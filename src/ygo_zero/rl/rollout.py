"""Async rollout collector for RL (PPO self-play, KL-anchored to a frozen BC policy).

Owns a VecEnv in async mode plus the per-env trajectory bookkeeping: each game is two
interleaved single-agent trajectories (one per seat), reward +1/-1 at the end (draw 0,
`trunc_reward` when cut by the decision cap). Episodes are either self-play (both seats
sampled from the learner, both trained) or, with probability `p_bc`, learner vs the frozen
BC policy (sampled, temperature 1; only the learner seat is stored).

One `ref` forward covers every returned row: BC-opponent rows sample their action from it,
learner rows keep its log-softmax as the cached KL reference (stored in fp16 by SlotStore).
"""

from __future__ import annotations

import time

import numpy as np
import torch

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
                 precision: str = "fp32") -> None:
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
        n = vec.num_envs
        self.buf = vec.buf
        self.traj: list[list[list[int]]] = [[[], []] for _ in range(n)]
        self.vs_bc = rng.random(n) < self.p_bc
        self.lseat = rng.integers(0, 2, n)
        vec.async_reset()

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
        t_env = t_fwd = 0.0
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
                if self.vs_bc[i]:
                    bc_games += 1
                    bc_wins += 0.5 if w not in (0, 1) else float(w == int(self.lseat[i]))
                self.vs_bc[i] = self.rng.random() < self.p_bc
                self.lseat[i] = self.rng.integers(0, 2)
            # Rows still in play (player -1 = idle schedule row: never sent again).
            player = buf["player"]
            ids = np.array([int(i) for i in ids if int(player[int(i)]) != -1], dtype=np.int64)
            if len(ids) == 0:
                if n_fin >= n_samples:
                    break
                continue
            actor = np.array([int(player[int(i)]) for i in ids], dtype=np.int64)
            is_learner = ~self.vs_bc[ids] | (actor == self.lseat[ids])
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
                if len(b_ids):
                    b_idx = np.array([pos[int(i)] for i in b_ids])
                    b_lp, b_legal = ref_lp_all[b_idx], legal_all[b_idx]
                    act_b, _ = _sample_from_lp(b_lp, b_legal)
                    for i, a in zip(b_ids, act_b):
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
        return finished, stats
