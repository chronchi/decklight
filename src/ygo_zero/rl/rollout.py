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
the batch, sampled temperature 1), or vs ygo-agent (`ygo(buf, rows, keys)` with the
Collector's own per-env episode tokens, since buf["game"] is -1 without a schedule).
The `ref` forward still covers every row so learner rows keep base as the KL anchor.
`league=None` keeps today's behaviour bit-identical.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ygo_zero.rl.league import BASE, SELF, YGO, League
from ygo_zero.rl.ppo import gather, masked_log_softmax, to_torch

TRUNCATED = 3  # VecEnv.TRUNCATED: game cut by set_max_decisions


@torch.no_grad()
def _sample_from_lp(lp: torch.Tensor, legal: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    p = lp.exp() * legal
    a = torch.multinomial(p, 1).squeeze(1)
    return (a.cpu().numpy().astype(np.int32),
            lp.gather(1, a[:, None]).squeeze(1).cpu().numpy())


class Collector:
    """VecEnv in async mode + per-env trajectory bookkeeping.

    Constructor calls `vec.async_reset()` and sends nothing yet: the first `collect`
    starts with `recv`. Envs stay in flight across `collect` calls (never re-reset).
    """

    def __init__(self, vec, model, ref, store, hist: int, known: bool, device: torch.device,
                 p_bc: float, trunc_reward: float, recv_batch: int, rng: np.random.Generator,
                 precision: str = "fp32", league: League | None = None, ygo=None) -> None:
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
        n = vec.num_envs
        self.buf = vec.buf
        self.traj: list[list[list[int]]] = [[[], []] for _ in range(n)]
        if league is None:
            self.vs_bc = rng.random(n) < self.p_bc
            self.lseat = rng.integers(0, 2, n)
        else:
            self.opp = np.array([league.sample() for _ in range(n)], dtype=object)
            self.lseat = rng.integers(0, 2, n)
            self.vs_bc = self.opp != SELF  # introspection only; episode draws go through league
            self.ep_id = np.arange(n, dtype=np.int64)  # per-env episode token for ygo keys
            self._ep_next = n
            self._net_cache: dict[str, object] = {}
            for name in set(self.opp.tolist()):
                if name not in (SELF, BASE, YGO):
                    self._net_cache[name] = league.get(name)
        vec.async_reset()

    def _draw_episode(self, i: int) -> None:
        """League path: record nothing here; draw the next episode type for env i."""
        self.opp[i] = self.league.sample()
        self.lseat[i] = self.rng.integers(0, 2)
        self.vs_bc[i] = self.opp[i] != SELF
        self.ep_id[i] = self._ep_next
        self._ep_next += 1
        name = str(self.opp[i])
        if name not in (SELF, BASE, YGO) and name not in self._net_cache:
            self._net_cache[name] = self.league.get(name)

    def _autocast(self):
        import contextlib

        if self.precision == "bf16" and self.device.type == "cuda":
            return torch.autocast(self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def collect(self, n_samples: int) -> tuple[list[tuple[list[int], float]], dict]:
        """Collect until `n_samples` finished samples are ready. Returns (finished, stats).

        `finished` is a list of (slots, reward), one per finished learner seat-trajectory.
        `stats` has the same per-iteration keys train_rl logs (games, trunc, bc_games, bc_wins,
        decisions, lengths) plus `t_env` (time blocked in recv) and `t_fwd` (forward+sample).
        """
        finished: list[tuple[list[int], float]] = []
        n_fin = 0
        games = trunc = bc_games = 0
        bc_wins = 0.0
        decisions = 0
        lengths: list[int] = []
        t_env = t_fwd = t_ga = 0.0
        lg = self.league is not None
        lg_games: dict[str, int] = {}
        lg_wins: dict[str, float] = {}
        opp_games = opp_learner_samples = 0
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
            for i in ids:
                i = int(i)
                if not bool(done[i]):
                    continue
                w = int(winner[i])
                games += 1
                trunc += w == TRUNCATED
                lengths.append(len(self.traj[i][0]) + len(self.traj[i][1]))
                n_ls = len(self.traj[i][int(self.lseat[i])]) if lg else 0
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
                        n_fin += len(sl)
                    self.traj[i][seat] = []
                if not lg:
                    if self.vs_bc[i]:
                        bc_games += 1
                        bc_wins += 0.5 if w not in (0, 1) else float(w == int(self.lseat[i]))
                    self.vs_bc[i] = self.rng.random() < self.p_bc
                    self.lseat[i] = self.rng.integers(0, 2)
                else:
                    res = 0.5 if w not in (0, 1) else float(w == int(self.lseat[i]))
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
                        elif name == YGO:
                            self.ygo.reset_keys([int(self.ep_id[i])])
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
            actor = np.array([int(player[int(i)]) for i in ids], dtype=np.int64)
            if not lg:
                is_learner = ~self.vs_bc[ids] | (actor == self.lseat[ids])
            else:
                is_learner = (self.opp[ids] == SELF) | (actor == self.lseat[ids])
            l_ids = ids[is_learner]
            b_ids = ids[~is_learner]
            t1 = time.perf_counter()
            with torch.inference_mode(), self._autocast():
                ob_all = gather(buf, ids, self.hist, self.known)
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
                    o_ids = b_ids[self.opp[b_ids] != BASE]
                    snap_rows = o_ids[self.opp[o_ids] != YGO]
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
                    if len(ga_rows):
                        t2 = time.perf_counter()
                        ga_keys = np.array([int(self.ep_id[int(i)]) for i in ga_rows])
                        act_g = self.ygo(buf, np.asarray(ga_rows), keys=ga_keys)
                        t_ga += time.perf_counter() - t2
                        for i, a in zip(ga_rows, act_g):
                            actions[pos[int(i)]] = int(a)
                if len(l_ids):
                    l_idx = torch.from_numpy(np.array([pos[int(i)] for i in l_ids], np.int64)).to(self.device)
                    tob_l = {k: v[l_idx] for k, v in tob_all.items()}
                    logits, v = self.model.forward_value(tob_l)
                    lp = masked_log_softmax(logits)
                    act_l, lp_l = _sample_from_lp(lp, torch.isfinite(logits))
                    ob_l = {k: v[np.array([pos[int(i)] for i in l_ids])] for k, v in ob_all.items()}
                    slots = self.store.put(ob_l, act_l, lp_l, v.float().cpu().numpy(),
                                           ref_lp_cpu[[pos[int(i)] for i in l_ids]])
                    for i, s in zip(l_ids, slots):
                        self.traj[int(i)][int(player[int(i)])].append(int(s))
                    for i, a in zip(l_ids, act_l):
                        actions[pos[int(i)]] = int(a)
            t_fwd += time.perf_counter() - t1
            self.vec.send(np.asarray(ids, dtype=np.int32), np.asarray(actions, dtype=np.int32))
            decisions += len(ids)
        stats = {"games": games, "trunc": trunc, "bc_games": bc_games, "bc_wins": bc_wins,
                 "decisions": decisions, "lengths": lengths, "t_env": t_env, "t_fwd": t_fwd}  # fmt: skip
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
