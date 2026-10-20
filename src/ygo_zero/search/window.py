"""Mid-game decision search (ADR 0006 M3): response windows (fire an interruption or hold) and own-turn plays.

States come from recorded games (any record with `config` + `choices`: ygo-agent teacher games, gold games, DAgger or
self-play records). The game is replayed; at each selected decision of player `me` the candidates are tried on engine
clones (replay of the prefix + the candidate) and each branch is played out by the policy for BOTH players (argmax)
until `horizon` turns have ended (1 = the end of the current turn). The candidate with the best position for `me`
becomes the decision's target; ties keep the policy's order (at a chain window PASS is ranked first, so firing must
earn its card).

Position score for `me`, default: field cards (mzone + szone) mine − theirs, + 0.5 x (hand mine − theirs),
+ (LP mine − theirs) / 2000; a finished duel scores ±50. With `value` (a learned value, ADR 0007 v2: P(the acting
player wins) in [-1, 1], e.g. `ValueNet.values`) the branch end is scored by the value from `me`'s view instead and a
finished duel scores ±1, so the target captures long-term effects a board count misses. The rollouts see the true hidden state (perfect
information): the target is "what would have worked here", which the student learns in expectation.

Selected decisions: `window` = a chain window (MSG_SELECT_CHAIN) on the opponent's turn where `me` can ACTIVATE;
`own` = an own-turn decision with >= 2 options from turn 2 on, sampled with probability `own_frac`.
Written as generators (see turn1.py) so many games share one batched forward (`run_many`).
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.eval.loopguard import LoopGuard
from ygo_zero.search.determinize import Determinizer, SeenTracker, key_rank_of_choice
from ygo_zero.search.turn1 import _batch, _masked, drive_many

LogitsFn = Callable[[dict[str, np.ndarray]], np.ndarray]
ValueFn = Callable[[dict[str, np.ndarray]], np.ndarray]  # batched obs -> value in [-1, 1], actor's view
WIN = 50.0


def position_score(obs, actor: int, me: int) -> float:
    """Score for player `me` from an obs seen by player `actor` (absolute seats)."""
    g = obs["global"]
    mine, theirs = (slice(6, 13), slice(13, 20)) if actor == me else (slice(13, 20), slice(6, 13))
    lp_m, lp_t = (int(g[0]), int(g[1])) if actor == me else (int(g[1]), int(g[0]))
    cm, ct = g[mine], g[theirs]
    return float((cm[2] + cm[3]) - (ct[2] + ct[3]) + 0.5 * (cm[1] - ct[1]) + (lp_m - lp_t) / 2000)


def is_window(obs, actor: int) -> bool:
    """A chain window on the opponent's turn where the acting player can activate something."""
    g = obs["global"]
    if int(g[21]) != C.MSG_SELECT_CHAIN or int(g[5]) == 1:
        return False
    acts, mask = obs["actions"], obs["action_mask"].astype(bool)
    return bool(np.any(acts[mask, 0] == Kind.ACTIVATE))


def decision_kind_all(obs, rng: random.Random, own_frac: float) -> str | None:
    """Stage L decision selection: "window" = a chain window (either turn) where an activation is legal; "own" =
    own-turn Main-phase idle decision with an activation legal, sampled with probability own_frac."""
    g = obs["global"]
    acts, mask = obs["actions"], obs["action_mask"].astype(bool)
    can_act = bool(np.any(acts[mask, 0] == Kind.ACTIVATE))
    if not can_act:
        return None
    if int(g[21]) == C.MSG_SELECT_CHAIN:
        return "window"
    if int(g[21]) == C.MSG_SELECT_IDLECMD and int(g[5]) == 1 and rng.random() < own_frac:
        return "own"
    return None


@dataclass
class WindowResult:
    targets: dict[int, int] = field(default_factory=dict)  # decision index -> searched option
    kinds: dict[int, str] = field(default_factory=dict)  # decision index -> "window" | "own"
    log: list[dict] = field(default_factory=list)
    replay_ok: bool = True
    sets: dict[int, list[int]] = field(default_factory=dict)  # decision index -> target set (option indices)
    conf: dict[int, float] = field(default_factory=dict)  # decision index -> confidence
    det_proposals: int = 0  # honest worlds proposed (rejection-sampling tries)
    det_accepted: int = 0  # ... accepted (root obs identical to the truth)
    det_fallbacks: int = 0  # ... rejected after `tries` (fell back to the true config)
    det_skipped: int = 0  # ... skipped searches (det_fallback="skip", no world accepted)
    det_s: float = 0.0  # wall seconds spent drawing honest worlds


def cards_score(obs, actor: int, me: int) -> float:
    """Raw card advantage for `me`: (hand + field) mine − theirs, from an obs seen by `actor`."""
    g = obs["global"]
    if actor == me:
        mine = int(g[7]) + int(g[8]) + int(g[9])
        theirs = int(g[14]) + int(g[15]) + int(g[16])
    else:
        mine = int(g[14]) + int(g[15]) + int(g[16])
        theirs = int(g[7]) + int(g[8]) + int(g[9])
    return float(mine - theirs)


BOARD_WIN = 50.0
VALUE_WIN = 1.0


class WindowSearch:
    def __init__(self, engine: Engine, logits: LogitsFn, k: int = 4, horizon: int = 1, own_frac: float = 0.0,
                 max_rollout: int = 300, batch: int = 512, seed: int = 0, value: ValueFn | None = None,
                 horizons: tuple[int, ...] | list[int] | None = None, samples: int = 1,
                  rollout_temp: float = 1.0, select: str | None = None, tie_eps: float = 0.0,
                  conf_skip: float = 1.01, ho=None, windows: str = "opp", prof: bool = False,
                  pool=None) -> None:  # fmt: skip
        self.engine, self.logits, self.k, self.horizon = engine, logits, k, horizon
        self.value, self.win = value, (1.0 if value is not None else WIN)
        self.own_frac, self.max_rollout, self.batch = own_frac, max_rollout, batch
        self.seed = seed
        self.horizons = tuple(horizons) if horizons is not None else None
        self.samples = samples
        self.rollout_temp = rollout_temp
        self.select = select
        self.tie_eps = tie_eps
        self.conf_skip = conf_skip  # game_gen skips decisions whose policy top probability is >= this (1.01 = never)
        self.capped = 0  # rollout branches that hit max_rollout (scored as a loss); gen_window prints it
        # Stage L (retrain plan v2): `ho` = search.honest.HonestRule -> decisive-label mode (merged candidates,
        # samples from `samples` up to ho.s_max in ho.step rounds, a target only where the best candidate beats the
        # runner-up significantly). `windows`: "opp" = opponent-turn chain windows + own_frac of own decisions (as
        # before); "all" = chain windows on either turn + own_frac of own Main-phase idle decisions with an activation.
        if windows not in ("opp", "all"):
            raise ValueError(f"windows must be 'opp' or 'all', got {windows!r}")
        self.ho, self.windows = ho, windows
        self.prof = prof  # --prof 1: rollouts_gen times each phase into self.last_prof (play is bit-identical)
        self.last_prof: dict = {}
        # Speed plan E6: a ThreadPoolExecutor for the branch-setup replays (None = today's
        # sequential loop, bit-identical). Set by scripts/search_match.py --replay-threads.
        self.pool = pool

    def _hs(self) -> tuple[int, ...]:
        return self.horizons if self.horizons is not None else (self.horizon,)

    def _select_key(self) -> str:
        if self.select is not None:
            return self.select
        h0 = self._hs()[0]
        return f"value@h{h0}" if self.value is not None else f"board@h{h0}"

    def _branch_envs(self, config: DuelConfig, auto_single: bool, branches,
                     worlds: list[DuelConfig | tuple[DuelConfig, list[int]]] | None):
        """Branch envs + root obs, in branch order (E6: with `self.pool`, the prefix
        replays run in parallel via `pool.map`, which keeps order, so the result is
        identical to the sequential loop)."""
        specs = []
        for p, _, _, s in branches:
            if worlds is None:
                cfg, pp = config, p
            else:
                w = worlds[s % len(worlds)]
                if isinstance(w, tuple):
                    cfg, wprefix = w
                    pp = list(wprefix) + [p[-1]]
                else:  # plain DuelConfig (tests): today's replay of p
                    cfg, pp = w, p
            specs.append((cfg, list(pp)))
        if self.pool is None:
            envs, obs = [], []
            for cfg, pp in specs:
                e = CppEnv(self.engine, cfg, auto_single)
                o = e.replay_from_reset(pp)
                envs.append(e)
                obs.append(o)
            return envs, obs

        def _one(spec):
            cfg, pp = spec
            e = CppEnv(self.engine, cfg, auto_single)
            return e, e.replay_from_reset(pp)

        pairs = list(self.pool.map(_one, specs))
        return [e for e, _ in pairs], [o for _, o in pairs]

    def _forward(self, obs_list: list[dict]) -> np.ndarray:
        out = [self.logits(_batch(obs_list[i : i + self.batch])) for i in range(0, len(obs_list), self.batch)]
        return np.concatenate(out)

    def _candidates(self, z: np.ndarray, obs, window: bool, merge: bool = False) -> list[int]:
        legal = np.flatnonzero(np.isfinite(z))
        order = [int(i) for i in legal[np.argsort(-z[legal], kind="stable")]]
        if merge:  # semantically identical options are one candidate (search/honest.py)
            from ygo_zero.search.honest import merge_candidates

            order = merge_candidates(order, obs)
        if not window:
            return order[: self.k]
        kinds = obs["actions"][:, 0]
        passes = [i for i in order if int(kinds[i]) == Kind.PASS]
        acts = [i for i in order if int(kinds[i]) == Kind.ACTIVATE]
        return (passes[:1] + acts)[: max(self.k, 2)]

    def rollouts_gen(self, config: DuelConfig, auto_single: bool, prefixes, me: int, end_turn: int = 0,
                        game_seed: int = 0, decision_k: int = 0, turn_start: int | None = None,
                        worlds: list[DuelConfig | tuple[DuelConfig, list[int]]] | None = None):
        """Play each branch out with the policy until the largest horizon ends (or the duel ends / max_rollout).

        `prefixes` is either the legacy list of choice prefixes (one branch per prefix, argmax rollouts,
        scored to `end_turn`) or a list of (prefix, cand_idx, cand_action, sample) tuples for the multi-score
        path. Returns `{score_key: [per-candidate averaged score]}` with keys `<name>@h<h>` for board / done /
        cards on every horizon plus value when `value` is set. Sample 0 of each candidate is argmax (as before);
        samples 1..S-1 sample from softmax(logits / rollout_temp) with a Random seeded from
        (seed, game_seed, k, candidate, sample). When the turn number first exceeds `turn_start + h - 1` the
        branch's scores for h are snapshotted; a branch that ended earlier keeps its terminal scores.
        Rollouts run under a LoopGuard (exact repeats within a turn mask the options already taken), and a branch
        that still hits max_rollout is scored as a loss for `me`: a candidate that sends play into a loop must not
        look as good as the position it froze in (round 1 learned such loops, NOTEBOOK 2026-10-06).
        With `worlds` (honest search), a list of `(world_config, world_prefix)` tuples, branch
        `(p, j, cand, s)` is built from `worlds[s % len(worlds)]` as `world_prefix + [p[-1]]`
        (p[-1] is the candidate; the root options are identical by the determinizer's obs check),
        so all candidates of sample s share one world (common random worlds). `None` = today's
        behaviour (bit-identical).
        """
        hs = self._hs()
        # Legacy call: list of prefixes -> one argmax branch each, single horizon ending at end_turn.
        if prefixes and isinstance(prefixes[0], list):
            turn_start_eff = end_turn - self.horizon + 1
            branches = [(list(p), j, -1, 0) for j, p in enumerate(prefixes)]
            single_legacy = True
        else:
            branches = [(list(p), int(j), int(a), int(s)) for p, j, a, s in prefixes]
            single_legacy = False
            turn_start_eff = turn_start if turn_start is not None else end_turn
        turn_start = turn_start_eff
        n_cands = max((j for _, j, _, _ in branches), default=-1) + 1
        B = len(branches)
        _prof = self.prof
        if _prof:
            _t0 = time.perf_counter()
        envs, obs = self._branch_envs(config, auto_single, branches, worlds)
        if _prof:
            _t_setup = time.perf_counter() - _t0
            _t_env = 0.0
            _t_wait = 0.0
            _t_value = 0.0
            _rounds = 0
            _obs_fwd = 0
            _t_roll0 = time.perf_counter()
        H = len(hs)
        snap_board: list[list[float | None]] = [[None] * H for _ in range(B)]
        snap_cards: list[list[float | None]] = [[None] * H for _ in range(B)]
        snap_done: list[list[float | None]] = [[None] * H for _ in range(B)]
        snap_actor: list[list[int]] = [[0] * H for _ in range(B)]
        snap_vobs: list[list[dict | None]] = [[None] * H for _ in range(B)]
        snap_vterm: list[list[float | None]] = [[None] * H for _ in range(B)]
        n = [0] * B
        live = [True] * B
        guard = LoopGuard()
        rngs = [random.Random(f"{self.seed}:{game_seed}:{decision_k}:{a}:{s}") for _, _, a, s in branches]

        def _snapshot(i: int, h_idx: int, terminal: bool, lost: bool = False) -> None:
            if terminal:
                w = 1 - me if lost else int(envs[i].winner)
                b = BOARD_WIN if w == me else -BOARD_WIN if w in (0, 1) else 0.0
                d = 1.0 if w == me else -1.0 if w in (0, 1) else 0.0
                snap_board[i][h_idx] = b
                snap_cards[i][h_idx] = 0.0
                snap_done[i][h_idx] = d
                snap_actor[i][h_idx] = me
                if self.value is not None:
                    snap_vterm[i][h_idx] = VALUE_WIN if w == me else -VALUE_WIN if w in (0, 1) else 0.0
            else:
                assert obs[i] is not None
                actor = int(envs[i].player)
                snap_board[i][h_idx] = position_score(obs[i], actor, me)
                snap_cards[i][h_idx] = cards_score(obs[i], actor, me)
                snap_done[i][h_idx] = 0.0
                snap_actor[i][h_idx] = actor
                if self.value is not None:
                    snap_vobs[i][h_idx] = obs[i]

        while any(live):
            act = []
            for i, e in enumerate(envs):
                if not live[i]:
                    continue
                if obs[i] is None:
                    for h_idx in range(H):
                        if snap_board[i][h_idx] is None:
                            _snapshot(i, h_idx, True)
                    live[i] = False
                    continue
                turn = int(obs[i]["global"][2])
                for h_idx, h in enumerate(hs):
                    if snap_board[i][h_idx] is None and turn > turn_start + h - 1:
                        _snapshot(i, h_idx, False)
                if all(v is not None for v in snap_board[i]):
                    live[i] = False
                    continue
                if n[i] >= self.max_rollout:
                    self.capped += 1
                    for h_idx in range(H):
                        if snap_board[i][h_idx] is None:
                            _snapshot(i, h_idx, True, lost=True)
                    live[i] = False
                    continue
                act.append(i)
            if not act:
                break
            if _prof:
                _tw = time.perf_counter()
            lg = yield [obs[i] for i in act]
            if _prof:
                _t_wait += time.perf_counter() - _tw
                _rounds += 1
                _obs_fwd += len(act)
            for i, row in zip(act, lg, strict=True):
                n[i] += 1
                z, key = guard.mask(i, obs[i], _masked(row, obs[i]))
                s = branches[i][3]
                if s == 0:
                    a = int(np.argmax(z))
                else:
                    finite = np.isfinite(z)
                    vals = z[finite] / self.rollout_temp
                    vals = vals - np.max(vals)
                    exp = np.exp(vals)
                    probs = exp / exp.sum()
                    r = rngs[i].random()
                    cum = 0.0
                    legal = np.flatnonzero(finite)
                    a = int(legal[-1])
                    for li, p in zip(legal, probs, strict=True):
                        cum += float(p)
                        if r < cum:
                            a = int(li)
                            break
                guard.note(i, key, a)
                if _prof:
                    _te = time.perf_counter()
                obs[i] = envs[i].step(a)
                if _prof:
                    _t_env += time.perf_counter() - _te
        # Any branch still live (should not happen) gets its current position for remaining horizons.
        for i in range(B):
            if live[i] and obs[i] is not None:
                for h_idx in range(H):
                    if snap_board[i][h_idx] is None:
                        _snapshot(i, h_idx, False)
            elif live[i]:
                for h_idx in range(H):
                    if snap_board[i][h_idx] is None:
                        _snapshot(i, h_idx, True)
        # Batched learned-value scoring for non-terminal snapshots.
        snap_value: list[list[float | None]] = [[None] * H for _ in range(B)]
        if _prof:
            _t_py = time.perf_counter() - _t_roll0 - _t_env - _t_wait
        if self.value is not None:
            for i in range(B):
                for h_idx in range(H):
                    if snap_vterm[i][h_idx] is not None:
                        snap_value[i][h_idx] = snap_vterm[i][h_idx]
            need = [(i, h_idx) for i in range(B) for h_idx in range(H)
                    if snap_board[i][h_idx] is not None and snap_value[i][h_idx] is None]
            if need:
                if _prof:
                    _tv = time.perf_counter()
                v = self.value(_batch([snap_vobs[i][h_idx] for i, h_idx in need]))  # type: ignore[index]
                if _prof:
                    _t_value += time.perf_counter() - _tv
                for (i, h_idx), x in zip(need, v, strict=True):
                    snap_value[i][h_idx] = float(x) if snap_actor[i][h_idx] == me else -float(x)
        # Average over samples per candidate.
        keys: list[str] = []
        for h in hs:
            keys.append(f"board@h{h}")
            if self.value is not None:
                keys.append(f"value@h{h}")
            keys.append(f"done@h{h}")
            keys.append(f"cards@h{h}")
        out: dict[str, list[float]] = {kk: [0.0] * n_cands for kk in keys}
        counts = [0] * n_cands
        for i, (_, j, _, _) in enumerate(branches):
            counts[j] += 1
            for h_idx, h in enumerate(hs):
                assert snap_board[i][h_idx] is not None and snap_cards[i][h_idx] is not None
                assert snap_done[i][h_idx] is not None
                out[f"board@h{h}"][j] += float(snap_board[i][h_idx])  # type: ignore[arg-type]
                out[f"done@h{h}"][j] += float(snap_done[i][h_idx])  # type: ignore[arg-type]
                out[f"cards@h{h}"][j] += float(snap_cards[i][h_idx])  # type: ignore[arg-type]
                if self.value is not None:
                    assert snap_value[i][h_idx] is not None
                    out[f"value@h{h}"][j] += float(snap_value[i][h_idx])  # type: ignore[arg-type]
        # Per-sample scores of the selection key, per candidate in branch order (play.py logs them as "samples": the
        # offline budget curve and per-decision noise SE need them; NOTEBOOK 2026-10-09 08:40). Read right after the call.
        sk = self._select_key()
        self.last_samples = [[] for _ in range(n_cands)]
        if sk in out and sk.startswith(("value@", "board@")):
            sh = hs.index(int(sk.split("@h")[1]))
            src = snap_value if sk.startswith("value@") else snap_board
            for i, (_, j, _, _) in enumerate(branches):
                self.last_samples[j].append(float(src[i][sh]))  # type: ignore[arg-type]
        # Per-sample scores of every score key, per candidate in branch order (fork_pilot.py reads
        # value@h1 / done@h99). Additive only: last_samples above is untouched.
        _src_of = {"board": snap_board, "cards": snap_cards, "done": snap_done}
        if self.value is not None:
            _src_of["value"] = snap_value
        self.last_samples_all: dict[str, list[list[float]]] = {
            kk: [[] for _ in range(n_cands)] for kk in keys
        }
        for kk in keys:
            name, h = kk.split("@h")
            h_idx = hs.index(int(h))
            src_all = _src_of[name]
            for i, (_, j, _, _) in enumerate(branches):
                self.last_samples_all[kk][j].append(float(src_all[i][h_idx]))  # type: ignore[arg-type]
        for kk in keys:
            for j in range(n_cands):
                if counts[j]:
                    out[kk][j] /= counts[j]
        if _prof:
            self.last_prof = {
                "t_setup": _t_setup, "t_env": _t_env, "t_py": _t_py, "t_wait": _t_wait,
                "rounds": _rounds, "obs_fwd": _obs_fwd, "rollout_len": list(n), "t_value": _t_value,
            }
        if single_legacy:
            # Legacy shape: plain score list for the single horizon (today's behaviour).
            sel = self._select_key()
            return [out[sel][j] for j in range(n_cands)]
        return out

    def game_gen(self, rec: dict, config: DuelConfig, game_seed: int = 0,
                 honest: bool = False, det_tries: int = 8, det_fallback: str = "true"):
        """Replay a recorded game and search its selected decisions. Returns a WindowResult.

        With `honest`, searched decisions roll out from determinized worlds (as in
        search/play.py `search_game_gen`): one SeenTracker per seat plus the replayed
        `choice_keys` feed a per-decision Determinizer whose worlds go to `rollouts_gen`.
        With `det_fallback="skip"` a decision with no accepted world is not searched.
        `honest=False` is today's behaviour (bit-identical).
        """
        if det_fallback not in ("true", "skip"):
            raise ValueError(f"det_fallback must be 'true' or 'skip', got {det_fallback!r}")
        rng = random.Random(self.seed * 1_000_003 + game_seed)
        auto_single = rec.get("auto_single", True)
        env = CppEnv(self.engine, config, auto_single)
        obs = env.reset()
        res = WindowResult()
        choices = rec["choices"]
        trackers = {0: SeenTracker(0), 1: SeenTracker(1)} if honest else None
        if trackers is not None:
            trackers[0].update(env)
            trackers[1].update(env)
        choice_keys: list = []
        for k, c in enumerate(choices):
            if obs is None:
                res.replay_ok = False
                break
            me, turn = int(env.player), int(obs["global"][2])
            n_legal = int(obs["action_mask"].sum())
            kind = None
            if self.windows == "all":
                kind = decision_kind_all(obs, rng, self.own_frac)
            elif is_window(obs, me):
                kind = "window"
            elif int(obs["global"][5]) == 1 and turn >= 2 and n_legal >= 2 and rng.random() < self.own_frac:
                kind = "own"
            if kind is not None and self.ho is not None:
                z = _masked((yield [obs])[0], obs)
                ez = np.exp(z - z.max())
                if float(ez.max() / ez.sum()) < self.conf_skip:
                    yield from self._decisive_gen(res, z, obs, kind, k, int(c), me, turn, choices, config, auto_single,
                                                  game_seed, trackers, choice_keys, det_tries, det_fallback)
            elif kind is not None:
                z = _masked((yield [obs])[0], obs)
                cands = self._candidates(z, obs, kind == "window")
                ez = np.exp(z - z.max())
                if len(cands) >= 2 and float(ez.max() / ez.sum()) < self.conf_skip:
                    sel = self._select_key()
                    branches = [(choices[:k] + [x], j, x, s) for j, x in enumerate(cands)
                                for s in range(self.samples)]
                    worlds = None
                    if trackers is not None:
                        det = Determinizer(
                            self.engine, config, auto_single, me, choices[:k],
                            trackers[me].seen, obs,
                            random.Random(f"{self.seed}:{game_seed}:{k}:det"),
                            prefix_keys=choice_keys[:k], tries=det_tries,
                        )
                        worlds = []
                        for _ in range(self.samples):
                            got = det.sample()
                            if got is None:
                                if det_fallback == "true":
                                    worlds.append((config, list(choices[:k])))
                            else:
                                worlds.append(got)
                        res.det_proposals += det.proposals
                        res.det_accepted += det.accepted
                        res.det_fallbacks += det.fallbacks
                        res.det_s += det.seconds
                        if det_fallback == "skip" and not worlds:
                            res.det_skipped += 1
                            worlds = None
                            choice_keys.append(key_rank_of_choice(obs, int(c)))
                            obs = env.step(int(c))
                            trackers[0].update(env)
                            trackers[1].update(env)
                            continue
                    scores_all = yield from self.rollouts_gen(
                        config, auto_single, branches, me, turn_start=turn,
                        game_seed=game_seed, decision_k=k, worlds=worlds)
                    assert isinstance(scores_all, dict)
                    sel_scores = scores_all[sel]
                    best = max(range(len(cands)), key=lambda j: (sel_scores[j], -j))
                    best_score = sel_scores[best]
                    # tie_eps 0: the single best (exact ties keep the candidate order, so firing must earn its card)
                    in_set = ([best] if self.tie_eps <= 0 else
                              [j for j in range(len(cands)) if sel_scores[j] >= best_score - self.tie_eps])
                    tset = [cands[j] for j in in_set]
                    in_set_s = set(in_set)
                    # confidence: share of the scores that separate the candidates whose top candidates (ties
                    # included) touch the target set; a flat score (e.g. done@h1 = 0 everywhere) does not vote
                    conf_keys = [kk for kk in scores_all if not kk.startswith("cards@")
                                 and max(scores_all[kk]) > min(scores_all[kk])]
                    agree = sum(1 for kk in conf_keys
                                if in_set_s & {j for j, x in enumerate(scores_all[kk]) if x == max(scores_all[kk])})
                    conf = agree / len(conf_keys) if conf_keys else 1.0
                    res.targets[k] = cands[best]
                    res.kinds[k] = kind
                    res.sets[k] = tset
                    res.conf[k] = conf
                    res.log.append({"k": k, "kind": kind, "turn": turn, "cands": cands,
                                    "scores": [round(s, 2) for s in sel_scores],
                                    "scores_all": {kk: [round(s, 2) for s in col] for kk, col in scores_all.items()},
                                    "set": tset, "conf": round(conf, 4),
                                    "chosen": cands[best],
                                    "played": int(c),
                                    "fired": int(obs["actions"][cands[best], 0]) == Kind.ACTIVATE})  # fmt: skip
            if trackers is not None:
                choice_keys.append(key_rank_of_choice(obs, int(c)))
            obs = env.step(int(c))
            if trackers is not None:
                trackers[0].update(env)
                trackers[1].update(env)
        return res

    def _decisive_gen(self, res: WindowResult, z: np.ndarray, obs, kind: str, k: int, played: int, me: int, turn: int,
                      choices, config: DuelConfig, auto_single: bool, game_seed: int, trackers, choice_keys,
                      det_tries: int, det_fallback: str):  # fmt: skip
        """Stage L decisive label for decision k (see __init__ `ho`); fills res.targets only on status "label"."""
        from ygo_zero.search.honest import decisive_label

        rule = self.ho
        cands = self._candidates(z, obs, kind == "window", merge=rule.merge)
        if len(cands) < 2:
            return
        acc: list[list[float]] = [[] for _ in cands]
        worlds_all: list | None = [] if trackers is not None else None
        det = None
        s_off, rounds, n_round = 0, 0, self.samples
        d = None
        while n_round > 0:
            if trackers is not None:
                if det is None:
                    det = Determinizer(self.engine, config, auto_single, me, choices[:k], trackers[me].seen, obs,
                                       random.Random(f"{self.seed}:{game_seed}:{k}:det"),
                                       prefix_keys=choice_keys[:k], tries=det_tries)  # fmt: skip
                new = []
                for _ in range(n_round):
                    got = det.sample()
                    if got is not None:
                        new.append(got)
                    elif det_fallback == "true":
                        new.append((config, list(choices[:k])))
                if not new:
                    break
                worlds_all.extend(new)
                n_new = len(new)
            else:
                n_new = n_round
            branches = [(choices[:k] + [x], j, x, s) for j, x in enumerate(cands) for s in range(s_off, s_off + n_new)]
            yield from self.rollouts_gen(config, auto_single, branches, me, turn_start=turn, game_seed=game_seed,
                                         decision_k=k, worlds=worlds_all)  # fmt: skip
            for j in range(len(cands)):
                acc[j].extend(self.last_samples[j])
            s_off += n_new
            rounds += 1
            d = decisive_label(acc, rule)
            if d.status == "label":
                break
            n_round = min(rule.step, rule.s_max - s_off)
        if det is not None:
            res.det_proposals += det.proposals
            res.det_accepted += det.accepted
            res.det_fallbacks += det.fallbacks
            res.det_s += det.seconds
        if d is None:
            res.det_skipped += 1
            return
        means = [float(np.mean(a)) if a else float("nan") for a in acc]
        target = cands[d.chosen_j] if d.status == "label" else None
        if target is not None:
            res.targets[k], res.kinds[k], res.sets[k], res.conf[k] = target, kind, [target], 1.0
        res.log.append({"k": k, "kind": kind, "turn": turn, "cands": cands, "scores": [round(m, 3) for m in means],
                        "chosen": target, "played": played,
                        "fired": target is not None and int(obs["actions"][target, 0]) == Kind.ACTIVATE,
                        "ho": {"status": d.status, "best": cands[d.chosen_j],
                               "runner_up": None if d.challenger_j is None else cands[d.challenger_j],
                               "gap": round(d.gap, 4), "se": round(d.se, 4) if math.isfinite(d.se) else None,
                               "n": d.n, "rounds": rounds}})  # fmt: skip

    def run_many(self, jobs, width: int = 32):
        """Drive (key, generator) jobs concurrently with one batched forward per step."""
        return drive_many(jobs, self._forward, width)
