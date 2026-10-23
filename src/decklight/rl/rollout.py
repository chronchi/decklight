"""Async rollout collector for RL (PPO self-play, KL-anchored to a frozen BC policy).

Owns a VecEnv in async mode plus the per-env trajectory bookkeeping: each game is two
interleaved single-agent trajectories (one per seat), reward +1/-1 at the end (draw 0,
`trunc_reward` when cut by the decision cap). Episodes are either self-play (both seats
sampled from the learner, both trained) or, with probability `p_bc`, learner vs the frozen
BC policy (sampled, temperature 1; only the learner seat is stored).

One `ref` forward covers every returned row: BC-opponent rows sample their action from it,
learner rows keep its log-softmax as the cached KL reference (stored in fp16 by SlotStore).

With a `league` (RL v2 part A), per-env episodes are drawn from the League mix instead:
self-play, vs base (as above), vs a frozen snapshot (one forward per snapshot present in
the batch, sampled temperature 1), vs a fixed checkpoint (same path as snapshots),
vs greedy (`greedy(buf, rows)`), or vs ygo-agent (`ygo(buf, rows, keys)` with the
Collector's own per-env episode tokens, since buf["game"] is -1 without a schedule).
The `ref` forward still covers every row so learner rows keep base as the KL anchor.
`league=None` keeps today's behaviour bit-identical.

Gauntlet-aware mode (`ga_share > 0`): the VecEnv draws gauntlet-vs-gauntlet episodes
itself (flagged in `buf["ga"]` with deck indices in `buf["deck0/1"]`) and the League
draw goes through `draw_ga`, so ygo-agent only plays gauntlet-vs-gauntlet and
pool-deck episodes never draw YGO. Finished YGO episodes whose decks miss the
gauntlet slice fail loud at the end of `collect` (`fail` callback or RuntimeError).
`ga_share = 0` keeps the league path bit-identical.

Mixed mode (`ga_mixed=True`, overfitting rule): the VecEnv draws one train-pool
deck and one gauntlet deck per gauntlet episode (`buf["ga"]` = 1/2 = seat holding
the gauntlet deck). The learner seat is then the non-gauntlet seat (never random),
and the gauntlet seat is never a learner seat: with SELF the gauntlet seat is
played by the current actor but its samples are not stored. ygo-agent still only
plays gauntlet episodes, on the gauntlet seat. `ga_mixed=False` (default) keeps
today's behaviour bit-identical.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from decklight.rl.league import BASE, GREEDY, SELF, YGO, League
from decklight.rl.critic import critic_obs
from decklight.rl.ppo import gather, masked_log_softmax, to_torch

TRUNCATED = 3  # VecEnv.TRUNCATED: game cut by set_max_decisions


@torch.no_grad()
def _sample_from_lp(lp: torch.Tensor, legal: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    p = lp.exp() * legal
    a = torch.multinomial(p, 1).squeeze(1)
    return (a.cpu().numpy().astype(np.int32),
            lp.gather(1, a[:, None]).squeeze(1).cpu().numpy())


def value_to_prob(v) -> np.ndarray:
    """Value-head output ([-1, 1] expected return) -> win probability in [0, 1].

    The head ends in Tanh (policy.py) and the PPO value loss is MSE against returns
    in [-1, 1] (+1 learner win / -1 loss), so E[return] = 2p - 1 and p = (v + 1) / 2.
    Clipped to [0, 1] for safety (fp16 store, non-finite guards elsewhere).
    """
    return np.clip((np.asarray(v, dtype=np.float64) + 1.0) / 2.0, 0.0, 1.0)


def episode_brier(values, won: float) -> tuple[float, float]:
    """Brier sums for one finished focus trajectory: (sse, open_sse).

    `values` are the stored per-step value-head outputs ([-1, 1]) in decision order,
    `won` is 1.0 (learner won) or 0.0 (lost); draws/truncated episodes never reach
    here (the caller skips w not in (0, 1)). `open_sse` is the single squared error
    of the episode's first learner decision (calibration before the game develops).
    """
    p = value_to_prob(values)
    se = (p - float(won)) ** 2
    return float(se.sum()), float(se[0]) if len(se) else 0.0


def dtype_entropy(counts: dict) -> float:
    """Entropy (nats) of a decision-type histogram {kind: count} (0.0 when empty).

    Kinds are the chosen option's `Kind` int (buf["actions"] row col 0, engine/actions.py).
    """
    tot = sum(counts.values())
    if not tot:
        return 0.0
    return float(-sum((c / tot) * math.log(c / tot) for c in counts.values() if c))


def summarize_lines(line_counts: dict) -> tuple[int, float]:
    """(distinct signatures, top share) from {signature: episodes} (0, 0.0 when empty)."""
    n = sum(line_counts.values())
    if not n:
        return 0, 0.0
    return len(line_counts), float(max(line_counts.values()) / n)


class Collector:
    """VecEnv in async mode + per-env trajectory bookkeeping.

    Constructor calls `vec.async_reset()` and sends nothing yet: the first `collect`
    starts with `recv`. Envs stay in flight across `collect` calls (never re-reset).
    """

    def __init__(self, vec, model, ref, store, hist: int, known: bool, device: torch.device,
                 p_bc: float, trunc_reward: float, recv_batch: int, rng: np.random.Generator,
                 precision: str = "fp32", league: League | None = None, ygo=None,
                 critic=None, priv_critic: bool = False, adv: str = "gae",
                 greedy=None, ga_share: float = 0.0, ga_range: tuple[int, int] | None = None,
                 ga_mixed: bool = False, fail=None, focus_idx: set[int] | None = None,
                 log_games: bool = False) -> None:
        self.vec = vec
        self.model = model
        self.ref = ref
        self.store = store
        self.hist = hist
        self.known = known
        self.device = torch.device(device)
        self.p_bc = float(p_bc)
        self.trunc_reward = float(trunc_reward)
        self.recv_batch = int(recv_batch)
        self.rng = rng
        self.precision = precision
        self.league = league
        self.ygo = ygo
        self.greedy = greedy
        self.ga_share = float(ga_share)
        self.ga_range = ga_range
        self.ga_mixed = bool(ga_mixed)
        self._fail = fail
        # Per-deck fine-tune (train_rl --focus-deck): indices into vec.decks holding the
        # focus deck. None keeps behaviour and stats bit-identical (no focus keys).
        self.focus_idx = set(focus_idx) if focus_idx is not None else None
        # Replay logging (train_rl --log-games): per-env choice/actor/turn/deal capture so
        # finished episodes are logged as replayable records (decks, seed, choices, ...).
        # False keeps behaviour bit-identical (no per-decision recording).
        self.log_games = bool(log_games)
        if self.ga_share < 0.0 or self.ga_share > 1.0:
            raise ValueError(f"ga_share={ga_share} not in [0, 1]")
        # Privileged critic (RL v2 part B): values come from `critic` on the privileged
        # obs; the policy forward then needs logits only. None keeps today's behaviour.
        if priv_critic and critic is None:
            raise ValueError("--priv-critic 1 needs a critic net")
        self.critic = critic
        self.priv_critic = bool(priv_critic)
        if adv not in ("gae", "vrpo"):
            raise ValueError(f"unknown adv {adv!r} (want 'gae' or 'vrpo')")
        self.adv = adv
        if adv == "vrpo":
            # Fail fast: the Q head must exist on the net the targets read it from.
            owner = critic if priv_critic else model
            if not getattr(owner, "has_q", False):
                raise ValueError("--adv vrpo needs a Q-capable critic "
                                 f"({'privileged critic' if priv_critic else 'policy net'} built with q=True)")
        n = vec.num_envs
        self.buf = vec.buf
        self.traj: list[list[list[int]]] = [[[], []] for _ in range(n)]
        # Replay-log capture of the episode now on each env (deck indices + deal seed,
        # taken at episode-begin sightings in _sync_start; -2 = not yet sighted).
        self._has_seed = "seed" in vec.buf
        self._has_decks = "deck0" in vec.buf and "deck1" in vec.buf
        self.ep_deck0 = np.full(n, -2, dtype=np.int64)
        self.ep_deck1 = np.full(n, -2, dtype=np.int64)
        self.ep_deal_seed = np.zeros(n, dtype=np.uint64)
        self.ep_choices: list[list[int]] = [[] for _ in range(n)]
        self.ep_actors: list[list[int]] = [[] for _ in range(n)]
        self.ep_turn = np.zeros(n, dtype=np.int64)
        # Diversity detail (only recorded with focus_idx set): per-env per-seat learner
        # decisions as (kind, code, turn) triples (kind/code = the taken option row's
        # cols 0/2 in buf["actions"], i.e. engine Kind + card code).
        self.ep_kinds: list[list[list[tuple[int, int, int]]]] = [[[], []] for _ in range(n)]
        # DAGS start index of the episode now on each env (-2 = not yet sighted; the C++
        # VecEnv starts the next episode before Python processes the previous finish, so the
        # index is captured at episode begin: on first sighting in play, and right after every
        # redraw, when buf["start"] already shows the new episode). Absent on buffers predating
        # the curriculum build: every episode then reports -1 (unknown, never recorded).
        self._has_start = "start" in vec.buf
        self.ep_start = np.full(n, -2, dtype=np.int64)
        self.ep_f0 = np.zeros(n, dtype=bool)
        self.ep_f1 = np.zeros(n, dtype=bool)
        if league is None:
            if self.ga_share > 0:
                raise ValueError("ga_share > 0 needs a league (deck-aware draws condition on buf['ga'])")
            if self.ga_mixed and ga_range is None:
                raise ValueError("ga_mixed needs ga_range=(start, count) for the fail-loud check")
            self.vs_bc = rng.random(n) < self.p_bc
            self.lseat = rng.integers(0, 2, n)
        elif self.ga_share <= 0:
            if self.ga_mixed and ga_range is None:
                raise ValueError("ga_mixed needs ga_range=(start, count) for the fail-loud check")
            self.opp = np.array([league.sample() for _ in range(n)], dtype=object)
            self.lseat = rng.integers(0, 2, n)
            self.vs_bc = self.opp != SELF  # introspection only; episode draws go through league
            self.ep_id = np.arange(n, dtype=np.int64)  # per-env episode token for ygo keys
            self._ep_next = n
            self._net_cache: dict[str, object] = {}
            for name in set(self.opp.tolist()):
                if name not in (SELF, BASE, YGO, GREEDY):
                    self._net_cache[name] = league.get(name)
            if league.p_greedy > 0 and greedy is None:
                raise ValueError("league p_greedy > 0 needs a greedy buf policy")
            self.ep_ga = np.zeros(n, dtype=np.int64)  # ga flag of the episode now on each env (0 = not ga)
        else:
            # Gauntlet-aware mode: the C++ VecEnv draws each episode's decks (gauntlet
            # vs pool, flagged in buf["ga"]) before Python sees it, so the initial
            # opponents cannot be drawn yet; each env draws on first sighting (before
            # its first decision is acted on) and on every later episode end, conditioned
            # on the actual current decks. Never draws YGO for a pool-deck episode.
            for k in ("ga", "deck0", "deck1"):
                if k not in vec.buf:
                    raise ValueError(f"ga_share > 0 needs buf[{k!r}] (rebuild the C++ env)")
            if ga_range is None:
                raise ValueError("ga_share > 0 needs ga_range=(start, count)")
            if league.p_greedy > 0 and greedy is None:
                raise ValueError("league p_greedy > 0 needs a greedy buf policy")
            if league.ga_ygo_share > 0 and ygo is None:
                raise ValueError("league ga_ygo_share > 0 needs the ygo-agent policy")
            self.opp = np.full(n, None, dtype=object)
            self.lseat = rng.integers(0, 2, n)
            self.vs_bc = np.zeros(n, dtype=bool)
            self.ep_id = np.arange(n, dtype=np.int64)
            self._ep_next = n
            self._net_cache = {}
            self.ep_d0 = np.full(n, -1, dtype=np.int64)  # deck indices of the episode now on each env
            self.ep_d1 = np.full(n, -1, dtype=np.int64)
            self.ep_ga = np.zeros(n, dtype=np.int64)  # ga flag (0/1 old, 0/1/2 mixed) of the episode now on each env
            self.ygo_bad = 0  # finished YGO episodes whose decks missed the gauntlet slice
        vec.async_reset()

    def _sync_start(self, i: int) -> None:
        """Capture buf["start"][i] as this env's current episode start (no-op without the key).

        Also captures the focus flags when focus_idx is set: buf["deck0/1"] already show the
        episode being drawn for (DAGS/scheduled episodes carry -1, never focus). Called at
        every episode-begin sighting, so at finish time the flags still describe the
        finished episode (the redraw happens after the finish is processed).
        The replay-log deck indices + deal seed (buf["deck0/1"], buf["seed"]) are captured
        the same way whenever the keys exist, so a finished non-DAGS episode is reproduced
        by DuelConfig(decks[d0], decks[d1], seed) + the logged choices.
        """
        if self._has_start:
            self.ep_start[i] = int(self.buf["start"][i])
        if self.focus_idx is not None and "deck0" in self.buf and "deck1" in self.buf:
            self.ep_f0[i] = int(self.buf["deck0"][i]) in self.focus_idx
            self.ep_f1[i] = int(self.buf["deck1"][i]) in self.focus_idx
        if self._has_decks:
            self.ep_deck0[i] = int(self.buf["deck0"][i])
            self.ep_deck1[i] = int(self.buf["deck1"][i])
        if self._has_seed:
            self.ep_deal_seed[i] = int(self.buf["seed"][i])

    def _opp_name(self, i: int) -> str:
        """train_rl --log-games `opponent` field for the episode now on env i."""
        if self.league is None:
            return "bc" if bool(self.vs_bc[i]) else "self"
        name = str(self.opp[i])
        if name == SELF:
            return "self"
        if name == BASE:
            return "bc"
        if name == YGO:
            return "ygoagent"
        if name == GREEDY:
            return "greedy"
        return f"pool:{name}"

    def _deck_name(self, d: int) -> str:
        """Deck list name for a logged record (falls back to the index)."""
        try:
            name = getattr(self.vec.decks[int(d)], "name", "")
        except (IndexError, AttributeError, TypeError):
            name = ""
        return name or f"deck{int(d)}"

    def _draw_episode(self, i: int) -> None:
        """League path: record nothing here; draw the next episode type for env i."""
        if self.ga_share <= 0:
            self.opp[i] = self.league.sample()
            self.lseat[i] = self.rng.integers(0, 2)
            self.vs_bc[i] = self.opp[i] != SELF
            self.ep_id[i] = self._ep_next
            self._ep_next += 1
            name = str(self.opp[i])
            if name not in (SELF, BASE, YGO, GREEDY) and name not in self._net_cache:
                self._net_cache[name] = self.league.get(name)
            if hasattr(self, "ep_ga"):
                self.ep_ga[i] = 0
            self._sync_start(i)  # buf already shows the episode just drawn for
            return
        self._draw_ga(i, first=False)

    def _draw_ga(self, i: int, first: bool) -> None:
        """Gauntlet-aware draw for env i, conditioned on its current episode's decks.

        `buf["ga"][i]` / `deck0/1[i]` already show the episode being drawn for (the
        VecEnv starts it before Python processes the previous finish; for `first`
        draws, before the first decision). `first` keeps the init episode token.
        Mixed mode (`ga_mixed`, overfitting rule): `buf["ga"]` is 1/2 = seat holding
        the gauntlet deck, and the learner seat is the other one (never random).
        """
        assert self.league is not None
        ga_flag = int(self.buf["ga"][i])  # 0/1 old (bool-backed), 0/1/2 mixed
        d0, d1 = int(self.buf["deck0"][i]), int(self.buf["deck1"][i])
        opp = self.league.draw_ga(bool(ga_flag))
        self.opp[i] = opp
        if self.ga_mixed and ga_flag in (1, 2):
            self.lseat[i] = 2 - ga_flag  # flag 1 (gauntlet seat 0) -> lseat 1, flag 2 -> lseat 0
        else:
            self.lseat[i] = self.rng.integers(0, 2)
        self.vs_bc[i] = opp != SELF
        if not first:
            self.ep_id[i] = self._ep_next
            self._ep_next += 1
        self.ep_d0[i], self.ep_d1[i] = d0, d1
        self.ep_ga[i] = ga_flag
        if opp not in (SELF, BASE, YGO, GREEDY) and opp not in self._net_cache:
            self._net_cache[opp] = self.league.get(opp)
        self._sync_start(i)  # buf already shows the episode just drawn for

    def _autocast(self):
        import contextlib

        if self.precision == "bf16" and self.device.type == "cuda":
            return torch.autocast(self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @staticmethod
    def _pad64(x: np.ndarray) -> np.ndarray:
        """Pad a [B, W<=64] row array to width 64 (policy pads logits to the full option count,
        which is 64 in the VecEnv; the store is fixed at 64 like ref_lp)."""
        if x.shape[1] == 64:
            return x
        assert x.shape[1] < 64, x.shape
        return np.pad(x, ((0, 0), (0, 64 - x.shape[1])))

    def _forward_learner_vrpo(self, tob_l, ob_l):
        """Learner forward for `--adv vrpo`: (logits, vbar_as_value, pi_cpu, q_cpu).

        pi is the behaviour policy's full per-option distribution (0 on illegal options);
        Q comes from the critic (privileged critic when `--priv-critic 1`, else the policy
        net's own Q head, single forward). The stored `value` is V̄(s) = Σ_a π Q.
        Non-finite values are stored as-is and fail loud in the batch targets.
        """
        if self.critic is None:
            logits, q = self.model.forward_q(tob_l)
        else:
            logits = self.model(tob_l)
            cobs_l = to_torch(critic_obs(ob_l, ob_l["priv_code"]), self.device)
            _, q = self.critic.forward_q(cobs_l)
        legal = torch.isfinite(logits)
        lp = masked_log_softmax(logits)
        pi = (lp.exp() * legal).float().cpu().numpy()
        q_cpu = q.float().cpu().numpy()
        vbar_rows = (pi * q_cpu).sum(-1)
        v = torch.from_numpy(vbar_rows.astype(np.float32)).to(logits.device)
        return logits, v, self._pad64(pi), self._pad64(q_cpu)

    def collect(self, n_samples: int) -> tuple[list[tuple[list[int], float]], dict]:
        """Collect until `n_samples` finished samples are ready. Returns (finished, stats).

        `finished` is a list of (slots, reward), one per finished learner seat-trajectory.
        `stats` has the same per-iteration keys train_rl logs (games, trunc, bc_games, bc_wins,
        decisions, lengths) plus `t_env` (time blocked in recv) and `t_fwd` (forward+sample).
        """
        finished: list[tuple[list[int], float]] = []
        fin_starts: list[int] = []  # starts-list index per finished trajectory (-1: not a DAGS episode)
        n_fin = 0
        games = trunc = bc_games = 0
        bc_wins = 0.0
        focus_traj = focus_traj_n = focus_bc_games = 0
        focus_bc_wins = 0.0
        # Per-deck fine-tune extras (only counted with focus_idx set): seat split of the
        # vs-BC focus games (seat 0 always moves first in the C++ VecEnv, so the learner
        # seat went first iff lseat == 0) and value-calibration sums over finished focus
        # trajectories (all opponents; draws/truncated excluded).
        focus_bc_first_games = focus_bc_second_games = 0
        focus_bc_first_wins = focus_bc_second_wins = 0.0
        focus_value_sse = focus_value_open_sse = 0.0
        focus_value_n = focus_value_open_n = 0
        # Play-diversity accumulators (only filled with focus_idx set): decision-type
        # histogram over learner decisions in finished focus trajectories, and first-turn
        # line signatures ({signature: episodes}; signature = tuple of the learner's
        # turn-1 (kind, code) pairs, only when the learner moves first, i.e. seat 0).
        focus_dtype_counts: dict[int, int] = {}
        focus_line_counts: dict[tuple, int] = {}
        focus_line_n = 0
        # Replay log (only filled with log_games=True): one replayable record per finished
        # non-DAGS episode (DAGS prefixes are mid-game starts, not reproducible from a deal).
        games_log: list[dict] = []
        dags_games = 0
        dags_wins = 0.0
        decisions = 0
        lengths: list[int] = []
        t_env = t_fwd = t_ga = 0.0
        lg = self.league is not None
        lg_games: dict[str, int] = {}
        lg_wins: dict[str, float] = {}
        opp_games = opp_learner_samples = 0
        priv_rows = priv_nonzero = 0
        ga_games = 0  # finished gauntlet episodes this collect (league + ga_share > 0)
        learner_ga = 0  # stored learner samples whose learner-seat deck is in the gauntlet slice
        n = self.vec.num_envs
        buf = self.buf
        while n_fin < n_samples:
            t0 = time.perf_counter()
            ids = np.asarray(self.vec.recv(self.recv_batch, n), dtype=np.int64)
            t_env += time.perf_counter() - t0
            if len(ids) == 0:
                continue
            # FIRST: finish episodes that just ended (their rows already show the next episode).
            done = buf["done"]
            winner = buf["winner"]
            dags_last = buf.get("dags_last")  # None on buffers predating DAGS: no episode is tagged
            for i in ids:
                i = int(i)
                if not bool(done[i]):
                    continue
                w = int(winner[i])
                games += 1
                trunc += w == TRUNCATED
                is_dags = bool(dags_last[i]) if dags_last is not None else False
                lengths.append(len(self.traj[i][0]) + len(self.traj[i][1]))
                n_ls = len(self.traj[i][int(self.lseat[i])]) if lg else 0
                fst = int(self.ep_start[i]) if self.ep_start[i] >= -1 else -1
                f0, f1 = bool(self.ep_f0[i]), bool(self.ep_f1[i])
                ls = int(self.lseat[i])
                for seat in (0, 1):
                    sl = self.traj[i][seat]
                    if w == TRUNCATED:
                        rew = self.trunc_reward
                    elif w == 2:
                        rew = 0.0
                    else:
                        rew = 1.0 if w == seat else -1.0
                    if sl:
                        finished.append((sl, rew))
                        # pass-rate slot = (start, seat): the two sides of a position are different problems
                        fin_starts.append(2 * fst + seat if fst >= 0 else -1)
                        n_fin += len(sl)
                        if self.focus_idx is not None:
                            focus_traj_n += 1
                            if (f0 if seat == 0 else f1):
                                focus_traj += 1
                                if w in (0, 1):  # draws/truncated excluded from calibration
                                    sse, open_sse = episode_brier(
                                        self.store.value[np.asarray(sl, dtype=np.int64)],
                                        1.0 if w == seat else 0.0)
                                    focus_value_sse += sse
                                    focus_value_n += len(sl)
                                    focus_value_open_sse += open_sse
                                    focus_value_open_n += 1
                                # Play diversity: ep_kinds holds this seat's learner decisions
                                # (non-empty exactly for stored learner seats); the seat went
                                # first iff seat == 0 (seat 0 always moves first).
                                kinds = self.ep_kinds[i][seat]
                                if kinds:
                                    for k, _, _ in kinds:
                                        focus_dtype_counts[k] = focus_dtype_counts.get(k, 0) + 1
                                    if seat == 0:
                                        sig = tuple((k, c) for k, c, t in kinds if t == 1)
                                        if sig:
                                            focus_line_counts[sig] = focus_line_counts.get(sig, 0) + 1
                                            focus_line_n += 1
                    self.traj[i][seat] = []
                if self.log_games and not is_dags:
                    d0, d1 = int(self.ep_deck0[i]), int(self.ep_deck1[i])
                    if d0 >= 0 and d1 >= 0 and self.ep_choices[i]:
                        games_log.append({
                            "decks": [self._deck_name(d0), self._deck_name(d1)],
                            "deck0": d0, "deck1": d1, "seed": int(self.ep_deal_seed[i]),
                            "choices": list(self.ep_choices[i]), "actors": list(self.ep_actors[i]),
                            "winner": w, "truncated": w == TRUNCATED,
                            "turns": int(self.ep_turn[i]),
                            "learner_seat": ls, "opponent": self._opp_name(i),
                            "focus": bool(f0 if ls == 0 else f1),
                        })
                self.ep_choices[i] = []
                self.ep_actors[i] = []
                self.ep_kinds[i] = [[], []]
                if not lg:
                    if self.vs_bc[i]:
                        bc_games += 1
                        bc_wins += 0.5 if w not in (0, 1) else float(w == ls)
                        if self.focus_idx is not None and (f0 if ls == 0 else f1):
                            focus_bc_games += 1
                            res_fbc = 0.5 if w not in (0, 1) else float(w == ls)
                            focus_bc_wins += res_fbc
                            if ls == 0:  # seat 0 always moves first: learner went first
                                focus_bc_first_games += 1
                                focus_bc_first_wins += res_fbc
                            else:
                                focus_bc_second_games += 1
                                focus_bc_second_wins += res_fbc
                    if is_dags:
                        dags_games += 1
                        dags_wins += 0.5 if w not in (0, 1) else float(w == int(self.lseat[i]))
                    self.vs_bc[i] = self.rng.random() < self.p_bc
                    self.lseat[i] = self.rng.integers(0, 2)
                    self._sync_start(i)  # buf already shows the episode just drawn for
                else:
                    res = 0.5 if w not in (0, 1) else float(w == int(self.lseat[i]))
                    if is_dags:
                        dags_games += 1
                        dags_wins += res
                    if self.opp[i] is None:
                        raise RuntimeError("episode finished before its first opponent draw")
                    if self.ga_share > 0 and hasattr(self, "ep_ga") and int(self.ep_ga[i]) != 0:
                        ga_games += 1
                    name = str(self.opp[i])
                    if name != SELF:
                        self.league.record(name, res)
                        lg_games[name] = lg_games.get(name, 0) + 1
                        lg_wins[name] = lg_wins.get(name, 0.0) + res
                        opp_games += 1
                        opp_learner_samples += n_ls
                        if name == BASE:
                            bc_games += 1
                            bc_wins += res
                            if self.focus_idx is not None and (f0 if ls == 0 else f1):
                                focus_bc_games += 1
                                focus_bc_wins += res
                                if ls == 0:  # seat 0 always moves first: learner went first
                                    focus_bc_first_games += 1
                                    focus_bc_first_wins += res
                                else:
                                    focus_bc_second_games += 1
                                    focus_bc_second_wins += res
                        elif name == YGO:
                            self.ygo.reset_keys([int(self.ep_id[i])])
                            if self.ga_share > 0:
                                assert self.ga_range is not None
                                lo, cnt = self.ga_range
                                d0, d1 = int(self.ep_d0[i]), int(self.ep_d1[i])
                                if self.ga_mixed:
                                    # Mixed layout: exactly one seat gauntlet, and it must be YGO's.
                                    ygo_seat = 1 - int(self.lseat[i])
                                    ygo_deck = d0 if ygo_seat == 0 else d1
                                    learn_deck = d1 if ygo_seat == 0 else d0
                                    if int(self.ep_ga[i]) == 0 or not (lo <= ygo_deck < lo + cnt) \
                                            or (lo <= learn_deck < lo + cnt):
                                        self.ygo_bad += 1
                                elif not (lo <= d0 < lo + cnt and lo <= d1 < lo + cnt):
                                    self.ygo_bad += 1
                        elif name == GREEDY:
                            pass  # stateless buf policy: nothing to reset
                    else:
                        lg_games[SELF] = lg_games.get(SELF, 0) + 1
                    self._draw_episode(i)
            if lg and self._net_cache:
                live = set(self.opp.tolist())
                for name in [k for k in self._net_cache if k not in live]:
                    del self._net_cache[name]
            # Rows still in play (player -1 = idle schedule row: never sent again).
            player = buf["player"]
            ids = np.array([int(i) for i in ids if int(player[int(i)]) != -1], dtype=np.int64)
            if len(ids) == 0:
                if n_fin >= n_samples:
                    break
                continue
            if lg and self.ga_share > 0:
                # Gauntlet-aware first sighting: draw before any row is acted on.
                for i in ids:
                    if self.opp[int(i)] is None:
                        self._draw_ga(int(i), first=True)
            # First sighting of any episode still in play: capture its start index
            # (initial episodes were drawn before Python ever saw buf["start"]).
            for i in ids:
                if self.ep_start[int(i)] == -2:
                    self._sync_start(int(i))
            actor = np.array([int(player[int(i)]) for i in ids], dtype=np.int64)
            if not lg:
                is_learner = ~self.vs_bc[ids] | (actor == self.lseat[ids])
            elif self.ga_mixed and self.ga_share > 0:
                # Overfitting rule: the gauntlet seat is never a learner seat, even
                # with SELF (then it is played by the current actor but not stored).
                ga_flags = np.asarray(buf["ga"][ids], dtype=np.int64)
                is_ga_ep = ga_flags != 0
                base = (self.opp[ids] == SELF) | (actor == self.lseat[ids])
                on_lseat = actor == self.lseat[ids]
                is_learner = np.where(is_ga_ep, on_lseat, base)
            else:
                is_learner = (self.opp[ids] == SELF) | (actor == self.lseat[ids])
            l_ids = ids[is_learner]
            b_ids = ids[~is_learner]
            t1 = time.perf_counter()
            with torch.inference_mode(), self._autocast():
                ob_all = gather(buf, ids, self.hist, self.known, priv=self.priv_critic)
                tob_all = to_torch(ob_all, self.device)
                ref_logits = self.ref(tob_all)
                ref_lp_all = masked_log_softmax(ref_logits)
                legal_all = torch.isfinite(ref_logits)
                ref_lp_cpu = ref_lp_all.float().cpu().numpy()
                actions = np.zeros(len(ids), np.int32)
                pos = {int(i): k for k, i in enumerate(ids)}
                bb_ids = b_ids
                if lg and len(b_ids):
                    bb_ids = b_ids[self.opp[b_ids] == BASE]
                if len(bb_ids):
                    b_idx = np.array([pos[int(i)] for i in bb_ids])
                    b_lp, b_legal = ref_lp_all[b_idx], legal_all[b_idx]
                    act_b, _ = _sample_from_lp(b_lp, b_legal)
                    for i, a in zip(bb_ids, act_b):
                        actions[pos[int(i)]] = int(a)
                if lg and len(b_ids):
                    o_ids = b_ids[(self.opp[b_ids] != BASE) & (self.opp[b_ids] != SELF)]
                    snap_rows = o_ids[(self.opp[o_ids] != YGO) & (self.opp[o_ids] != GREEDY)]
                    for name in set(self.opp[snap_rows].tolist()) if len(snap_rows) else ():
                        s_rows = snap_rows[self.opp[snap_rows] == name]
                        s_idx = torch.from_numpy(
                            np.array([pos[int(i)] for i in s_rows], np.int64)).to(self.device)
                        tob_s = {k: v[s_idx] for k, v in tob_all.items()}
                        s_logits = self._net_cache[name](tob_s)
                        s_lp = masked_log_softmax(s_logits)
                        act_s, _ = _sample_from_lp(s_lp, torch.isfinite(s_logits))
                        for i, a in zip(s_rows, act_s):
                            actions[pos[int(i)]] = int(a)
                    ga_rows = o_ids[self.opp[o_ids] == YGO]
                    if len(ga_rows) and self.ga_share > 0:  # live check: the episode ygo-agent plays now
                        assert self.ga_range is not None
                        lo, cnt = self.ga_range
                        if self.ga_mixed:
                            # Mixed layout: YGO's seat (1 - lseat) must hold the gauntlet deck.
                            for _i in np.asarray(ga_rows):
                                _ii = int(_i)
                                _ygo_seat = 1 - int(self.lseat[_ii])
                                _dd0, _dd1 = int(buf["deck0"][_ii]), int(buf["deck1"][_ii])
                                _ygo_deck = _dd0 if _ygo_seat == 0 else _dd1
                                _learn_deck = _dd1 if _ygo_seat == 0 else _dd0
                                if int(buf["ga"][_ii]) == 0 or not (lo <= _ygo_deck < lo + cnt) \
                                        or (lo <= _learn_deck < lo + cnt):
                                    self.ygo_bad += 1
                        else:
                            d = np.stack([buf["deck0"][ga_rows], buf["deck1"][ga_rows]])
                            self.ygo_bad += int((((d < lo) | (d >= lo + cnt)).any(0)).sum())
                    if len(ga_rows):
                        t2 = time.perf_counter()
                        ga_keys = np.array([int(self.ep_id[int(i)]) for i in ga_rows])
                        act_g = self.ygo(buf, np.asarray(ga_rows), keys=ga_keys)
                        t_ga += time.perf_counter() - t2
                        for i, a in zip(ga_rows, act_g):
                            actions[pos[int(i)]] = int(a)
                    gr_rows = o_ids[self.opp[o_ids] == GREEDY]
                    if len(gr_rows):
                        if self.greedy is None:
                            raise RuntimeError("league drew 'greedy' with no greedy policy")
                        act_gr = self.greedy(buf, np.asarray(gr_rows))
                        for i, a in zip(gr_rows, act_gr):
                            actions[pos[int(i)]] = int(a)
                # Mixed SELF gauntlet seats: played by the current actor but never stored
                # (old mode: SELF episodes have no non-learner rows, so this is empty).
                if lg and len(b_ids):
                    _self_b = b_ids[self.opp[b_ids] == SELF]
                else:
                    _self_b = np.zeros(0, dtype=np.int64)
                if len(l_ids) or len(_self_b):
                    _m_ids = np.concatenate([np.asarray(l_ids), np.asarray(_self_b)]) \
                        if len(l_ids) and len(_self_b) else (np.asarray(l_ids) if len(l_ids)
                                                             else np.asarray(_self_b))
                    l_idx = torch.from_numpy(np.array([pos[int(i)] for i in _m_ids], np.int64)).to(self.device)
                    tob_l = {k: v[l_idx] for k, v in tob_all.items() if k != "priv_code"}
                    ob_l = {k: v[np.array([pos[int(i)] for i in _m_ids])] for k, v in ob_all.items()}
                    if self.adv == "vrpo":
                        logits, v, pi_cpu, q_cpu = self._forward_learner_vrpo(tob_l, ob_l)
                    elif self.critic is None:
                        logits, v = self.model.forward_value(tob_l)
                        pi_cpu = q_cpu = None
                    else:
                        logits = self.model(tob_l)
                        cobs_l = to_torch(critic_obs(ob_l, ob_l["priv_code"]), self.device)
                        _, v = self.critic.forward_value(cobs_l)
                        pi_cpu = q_cpu = None
                    lp = masked_log_softmax(logits)
                    act_m, lp_m = _sample_from_lp(lp, torch.isfinite(logits))
                    _n_l = len(l_ids)
                    act_l, lp_l = act_m[:_n_l], lp_m[:_n_l]
                    act_self = act_m[_n_l:]
                    if _n_l:
                        slots = self.store.put(
                            {k: v[:_n_l] for k, v in ob_l.items()}, act_l, lp_l,
                            v.float().cpu().numpy()[:_n_l],
                            ref_lp_cpu[[pos[int(i)] for i in l_ids]],
                            pi=pi_cpu[:_n_l] if pi_cpu is not None else None,
                            q=q_cpu[:_n_l] if q_cpu is not None else None)
                    if self.priv_critic:
                        priv_rows += len(l_ids)
                        priv_nonzero += int((ob_l["priv_code"][:_n_l] > 0).sum()) if _n_l else 0
                    if self.ga_range is not None and _n_l:
                        # Fail-loud input for the overfitting rule: learner rows on gauntlet decks.
                        _lo, _cnt = self.ga_range
                        _d0 = np.asarray(buf["deck0"][np.asarray(l_ids)], dtype=np.int64)
                        _d1 = np.asarray(buf["deck1"][np.asarray(l_ids)], dtype=np.int64)
                        _seats = np.asarray(player[np.asarray(l_ids)], dtype=np.int64)
                        _decks = np.where(_seats == 0, _d0, _d1)
                        learner_ga += int((((_decks >= _lo) & (_decks < _lo + _cnt))).sum())
                    if _n_l:
                        for i, s in zip(l_ids, slots):
                            self.traj[int(i)][int(player[int(i)])].append(int(s))
                        for i, a in zip(l_ids, act_l):
                            actions[pos[int(i)]] = int(a)
                    for i, a in zip(np.asarray(_self_b).tolist(), act_self.tolist()):
                        actions[pos[int(i)]] = int(a)
            t_fwd += time.perf_counter() - t1
            if self.log_games or self.focus_idx is not None:
                # Per-decision replay/diversity capture (buf rows are valid until sent again).
                # kinds/codes come from the taken option's row (cols 0/2) in buf["actions"].
                _acts = buf["actions"] if self.focus_idx is not None else None
                _glob = buf["global"]
                for k, i in enumerate(ids):
                    ii = int(i)
                    a = int(actions[pos[ii]])
                    if self.log_games:
                        self.ep_choices[ii].append(a)
                        self.ep_actors[ii].append(int(player[ii]))
                        self.ep_turn[ii] = int(_glob[ii][2])
                if self.focus_idx is not None and len(l_ids):
                    for i, a in zip(l_ids, act_l):
                        ii, aa = int(i), int(a)
                        seat = int(player[ii])
                        assert _acts is not None
                        self.ep_kinds[ii][seat].append(
                            (int(_acts[ii, aa, 0]), int(_acts[ii, aa, 2]), int(_glob[ii][2])))
            self.vec.send(np.asarray(ids, dtype=np.int32), np.asarray(actions, dtype=np.int32))
            decisions += len(ids)
        if lg and self.ga_share > 0 and self.ygo_bad:
            msg = (f"{self.ygo_bad} ygo-agent episodes with decks outside the gauntlet slice "
                   f"{self.ga_range} (deck-aware league invariant broken)")
            line = f"ALERT rollout: {msg}"
            print(line, flush=True)
            if self._fail is not None:
                self._fail(msg)
            else:
                raise RuntimeError(line)
        stats = {"games": games, "trunc": trunc, "bc_games": bc_games, "bc_wins": bc_wins,
                 "decisions": decisions, "lengths": lengths, "t_env": t_env, "t_fwd": t_fwd,
                 "dags_games": dags_games,
                 "dags_wr": round(dags_wins / max(dags_games, 1), 3),
                 "dags_bad": int(getattr(self.vec, "dags_bad", 0)),
                 "fin_starts": fin_starts,
                 "ga_games": ga_games, "learner_ga": learner_ga}  # fmt: skip
        if self.focus_idx is not None:
            _line_distinct, _line_top_share = summarize_lines(focus_line_counts)
            stats.update(focus_traj=focus_traj, focus_traj_n=focus_traj_n,
                         focus_bc_games=focus_bc_games, focus_bc_wins=focus_bc_wins,
                         focus_bc_first_games=focus_bc_first_games,
                         focus_bc_first_wins=focus_bc_first_wins,
                         focus_bc_second_games=focus_bc_second_games,
                         focus_bc_second_wins=focus_bc_second_wins,
                         focus_value_sse=focus_value_sse, focus_value_n=focus_value_n,
                         focus_value_open_sse=focus_value_open_sse,
                         focus_value_open_n=focus_value_open_n,
                         focus_dtype_entropy=dtype_entropy(focus_dtype_counts),
                         focus_dtype_n=sum(focus_dtype_counts.values()),
                         focus_line_distinct=_line_distinct,
                         focus_line_top=(max(focus_line_counts.values())
                                         if focus_line_counts else 0),
                         focus_line_n=focus_line_n,
                         focus_line_top_share=_line_top_share)
        stats["games_log"] = games_log
        if self.priv_critic:
            stats["priv_rows"] = priv_rows
            stats["priv_nonzero"] = priv_nonzero
        if lg:
            assert self.league is not None
            probs = self.league.sampling_probs()
            league_stats = {}
            for name in probs:
                g = lg_games.get(name, 0)
                league_stats[name] = {
                    "games": g,
                    "wr": round(lg_wins.get(name, 0.0) / max(g, 1), 3),
                    "pfsp_w": round(probs[name], 4),
                }
            stats["league"] = league_stats
            stats["t_ga"] = t_ga
            stats["opp_games"] = opp_games
            stats["opp_learner_samples"] = opp_learner_samples
        return finished, stats
