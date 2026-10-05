"""PPO pieces for RL v0 (PLAN Phase 5): sample storage, per-seat trajectories, GAE, the loss.

A game is two interleaved single-agent trajectories, one per seat: each seat's own decisions in order, reward
+1 / -1 at the end (draw 0, cut by the decision cap: `trunc_reward`), no discount. Samples live in a preallocated
slot store so open trajectories survive across PPO iterations (their stored log-probs then come from an older
policy: the PPO ratio corrects for that).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ygo_zero.model.bc_data import KNOWN_KEYS, OBS_KEYS


def model_keys(hist: int, known: bool) -> tuple[str, ...]:
    """Obs keys a PolicyNet with this config reads from a VecEnv buffer (history comes from hist_long if hist>0)."""
    keys = tuple(OBS_KEYS) if hist == 0 else tuple(k for k in OBS_KEYS if k not in ("history", "history_mask"))
    keys += ("hist_long", "hist_long_mask") if hist > 0 else ()
    return keys + (tuple(KNOWN_KEYS) if known else ())


def gather(buf: dict[str, np.ndarray], rows: np.ndarray, hist: int, known: bool) -> dict[str, np.ndarray]:
    """Rows of the buffer as the model's input dict (copies; hist_long already cut to `hist`)."""
    out = {}
    for k in model_keys(hist, known):
        if k == "hist_long":
            out["history"] = buf[k][rows, :hist]
        elif k == "hist_long_mask":
            out["history_mask"] = buf[k][rows, :hist]
        else:
            out[k] = buf[k][rows]
    return out


def to_torch(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in obs.items()}


class SlotStore:
    """Fixed-capacity sample store: obs arrays + action, old log-prob, old value; slots handed out and freed."""

    def __init__(self, capacity: int, example: dict[str, np.ndarray]) -> None:
        self.obs = {k: np.zeros((capacity,) + v.shape[1:], v.dtype) for k, v in example.items()}
        self.action = np.zeros(capacity, np.int64)
        self.logp = np.zeros(capacity, np.float32)
        self.value = np.zeros(capacity, np.float32)
        self._free = list(range(capacity - 1, -1, -1))

    @property
    def free(self) -> int:
        return len(self._free)

    def put(self, obs: dict[str, np.ndarray], action, logp, value) -> np.ndarray:
        n = len(action)
        if n > len(self._free):
            raise MemoryError("slot store full: raise --store or lower --rollout")
        slots = np.array([self._free.pop() for _ in range(n)], np.int64)
        for k, v in obs.items():
            self.obs[k][slots] = v
        self.action[slots], self.logp[slots], self.value[slots] = action, logp, value
        return slots

    def release(self, slots) -> None:
        self._free.extend(int(s) for s in slots)


def gae(values: np.ndarray, reward: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """One finished seat trajectory, gamma = 1, reward only at the end. Returns (advantages, returns)."""
    n = len(values)
    nxt = np.append(values[1:], 0.0).astype(np.float32)
    delta = nxt - values
    delta[-1] += reward
    adv = np.zeros(n, np.float32)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        acc = delta[t] + lam * acc
        adv[t] = acc
    return adv, adv + values


def masked_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """log-softmax over legal options (illegal logits are -inf); illegal entries come out as 0, not nan."""
    lp = F.log_softmax(logits.float(), -1)
    return torch.where(torch.isfinite(logits), lp, torch.zeros_like(lp))


def ppo_loss(model, ref, obs, action, old_logp, adv, ret, clip: float, vf_coef: float, kl_coef: float,
             ent_coef: float) -> tuple[torch.Tensor, dict]:  # fmt: skip
    logits, value = model.forward_value(obs)
    lp_all = masked_log_softmax(logits)
    legal = torch.isfinite(logits)
    p = lp_all.exp() * legal
    logp = lp_all.gather(1, action[:, None]).squeeze(1)
    ratio = torch.exp(logp - old_logp)
    pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    vf = F.mse_loss(value.float(), ret)
    ent = -(p * lp_all).sum(1).mean()
    loss = pg + vf_coef * vf - ent_coef * ent
    kl = torch.zeros((), device=logits.device)
    if ref is not None and kl_coef > 0:
        with torch.no_grad():
            ref_lp = masked_log_softmax(ref(obs))
        kl = (p * (lp_all - ref_lp)).sum(1).mean()  # KL(pi || pi_BC), exact over the legal options
        loss = loss + kl_coef * kl
    stats = {
        "pg": pg.item(), "vf": vf.item(), "ent": ent.item(), "kl_bc": kl.item(),
        "clipfrac": ((ratio - 1).abs() > clip).float().mean().item(),
        "approx_kl": ((ratio - 1) - torch.log(ratio)).mean().item(),
    }  # fmt: skip
    return loss, stats
