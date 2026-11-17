"""PPO pieces for RL v0 (PLAN Phase 5): sample storage, per-seat trajectories, GAE, the loss.

A game is two interleaved single-agent trajectories, one per seat: each seat's own decisions in order, reward
+1 / -1 at the end (draw 0, cut by the decision cap: `trunc_reward`), no discount. Samples live in a preallocated
slot store so open trajectories survive across PPO iterations (their stored log-probs then come from an older
policy: the PPO ratio corrects for that).
"""

from __future__ import annotations

import threading

import numpy as np
import torch
import torch.nn.functional as F

from decklight.model.bc_data import KNOWN_KEYS, OBS_HISTORY_KEYS, OBS_KEYS


def model_keys(hist: int, known: bool, observable: bool = False) -> tuple[str, ...]:
    """Obs keys a PolicyNet with this config reads from a VecEnv buffer (history comes from hist_long if hist>0;
    an observable-history net also reads obs_history/obs_history_mask)."""
    keys = tuple(OBS_KEYS) if hist == 0 else tuple(k for k in OBS_KEYS if k not in ("history", "history_mask"))
    keys += ("hist_long", "hist_long_mask") if hist > 0 else ()
    keys += tuple(OBS_HISTORY_KEYS) if observable else ()
    return keys + (tuple(KNOWN_KEYS) if known else ())


def gather(buf: dict[str, np.ndarray], rows: np.ndarray, hist: int, known: bool,
           priv: bool = False, observable: bool = False) -> dict[str, np.ndarray]:
    """Rows of the buffer as the model's input dict (copies; hist_long already cut to `hist`).

    `priv=True` also returns `priv_code` (the true codes of hidden rows, for the
    privileged critic); the default False keeps the policy path bit-identical.
    """
    out = {}
    for k in model_keys(hist, known, observable):
        if k == "hist_long":
            out["history"] = buf[k][rows, :hist]
        elif k == "hist_long_mask":
            out["history_mask"] = buf[k][rows, :hist]
        else:
            out[k] = buf[k][rows]
    if priv:
        out["priv_code"] = buf["priv_code"][rows]
    return out


def to_torch(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in obs.items()}


class SlotStore:
    """Fixed-capacity sample store: obs arrays + action, old log-prob, old value; slots handed out and freed.

    VRPO (`vrpo=True`) additionally stores the behaviour policy's full per-option
    probabilities (`pi`, fp16 like `ref_lp`) and the critic's Q(s,.) (`q`, fp16).
    `vrpo=False` (default) keeps today's layout and behaviour bit-identical.
    """

    def __init__(self, capacity: int, example: dict[str, np.ndarray], vrpo: bool = False) -> None:
        self.obs = {k: np.zeros((capacity,) + v.shape[1:], v.dtype) for k, v in example.items()}
        self.action = np.zeros(capacity, np.int64)
        self.logp = np.zeros(capacity, np.float32)
        self.value = np.zeros(capacity, np.float32)
        self.isw = np.ones(capacity, np.float32)  # min(1, pi/mu) for --win-explore (1.0 elsewhere)
        self.ref_lp = np.zeros((capacity, 64), np.float16)
        self.vrpo = bool(vrpo)
        self.pi: np.ndarray | None = np.zeros((capacity, 64), np.float16) if vrpo else None
        self.q: np.ndarray | None = np.zeros((capacity, 64), np.float16) if vrpo else None
        self._free = list(range(capacity - 1, -1, -1))
        self._lock = threading.Lock()

    @property
    def free(self) -> int:
        with self._lock:
            return len(self._free)

    def put(self, obs: dict[str, np.ndarray], action, logp, value, ref_lp=None,
            pi=None, q=None, isw=None) -> np.ndarray:
        n = len(action)
        with self._lock:
            if n > len(self._free):
                raise MemoryError("slot store full: raise --store or lower --rollout")
            slots = np.array([self._free.pop() for _ in range(n)], np.int64)
        for k, v in obs.items():
            self.obs[k][slots] = v
        self.action[slots], self.logp[slots], self.value[slots] = action, logp, value
        if isw is None:
            self.isw[slots] = 1.0
        else:
            self.isw[slots] = np.asarray(isw, dtype=np.float32)
        if ref_lp is None:
            self.ref_lp[slots] = 0
        else:
            self.ref_lp[slots] = np.asarray(ref_lp, dtype=np.float16)
        if pi is not None or q is not None:
            if self.pi is None or self.q is None:
                cap = self.action.shape[0]
                self.pi = np.zeros((cap, 64), np.float16)
                self.q = np.zeros((cap, 64), np.float16)
            assert self.pi is not None and self.q is not None
            if pi is None:
                self.pi[slots] = 0
            else:
                self.pi[slots] = np.asarray(pi, dtype=np.float16)
            if q is None:
                self.q[slots] = 0
            else:
                self.q[slots] = np.asarray(q, dtype=np.float16)
        return slots

    def release(self, slots) -> None:
        with self._lock:
            self._free.extend(int(s) for s in slots)


def gae(values: np.ndarray, reward: float, lam: float,
        step_rewards: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One finished seat trajectory, gamma = 1, reward only at the end. Returns (advantages, returns).

    `step_rewards` (optional, one per decision) adds r_t to the TD error at step t
    (used for potential-based shaping; the last entry must include the terminal
    potential). None is today's code path, bit-identical.
    """
    n = len(values)
    nxt = np.append(values[1:], 0.0).astype(np.float32)
    delta = nxt - values
    delta[-1] += reward
    if step_rewards is not None:
        delta += np.asarray(step_rewards, dtype=np.float32)
    adv = np.zeros(n, np.float32)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        acc = delta[t] + lam * acc
        adv[t] = acc
    return adv, adv + values


def opp_shape_potential(glob: np.ndarray, lam: float) -> np.ndarray:
    """Potential Phi(s) = -lam * (opp hand + opp monsters + opp spell/trap) / 10.

    Columns 14/15/16 of obs["global"] (the learner's view) are the opponent's hand,
    monster-zone and spell/trap-zone counts.
    """
    g = np.asarray(glob)
    cards = (g[:, 14].astype(np.float32) + g[:, 15].astype(np.float32)
             + g[:, 16].astype(np.float32))
    return (-float(lam) * cards / 10.0).astype(np.float32)


def masked_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """log-softmax over legal options (illegal logits are -inf); illegal entries come out as 0, not nan."""
    lp = F.log_softmax(logits.float(), -1)
    return torch.where(torch.isfinite(logits), lp, torch.zeros_like(lp))


def explained_variance(value: torch.Tensor, ret: torch.Tensor) -> torch.Tensor:
    """1 - Var(ret - value) / Var(ret): 1 = perfect, 0 = mean predictor (detached 0-d)."""
    var_ret = ret.float().var(unbiased=False)
    ev = 1.0 - (ret.float() - value.float()).var(unbiased=False) / (var_ret + 1e-8)
    return ev.detach()


def ppo_loss_priv(policy, critic, ref_lp, obs, cobs, action, old_logp, adv, ret, clip: float,
                   vf_coef: float, kl_coef: float, ent_coef: float,
                   isw: torch.Tensor | None = None,
                   ) -> tuple[torch.Tensor, torch.Tensor, dict]:  # fmt: skip
    """PPO with a privileged critic: the policy loss (pg + KL - entropy) trains the
    policy from its own obs, the value loss trains the critic from the privileged
    obs. Returns (policy_loss, critic_loss, stats); stats match `ppo_loss` plus
    `critic_ev` / `policy_ev` (explained variance of each value function on returns)."""
    logits, pval = policy.forward_value(obs)
    _, cval = critic.forward_value(cobs)
    lp_all = masked_log_softmax(logits)
    legal = torch.isfinite(logits)
    p = lp_all.exp() * legal
    logp = lp_all.gather(1, action[:, None]).squeeze(1)
    ratio = torch.exp(logp - old_logp)
    if isw is not None:
        adv = adv * isw.to(dtype=adv.dtype)
    pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    vf = F.mse_loss(cval.float(), ret)
    ent = -(p * lp_all).sum(1).mean()
    loss_pol = pg - ent_coef * ent
    loss_crit = vf_coef * vf
    kl = torch.zeros((), device=logits.device)
    if ref_lp is not None and kl_coef > 0:
        ref_lp = ref_lp.to(dtype=torch.float32) if torch.is_tensor(ref_lp) else torch.as_tensor(ref_lp)
        ref_lp = ref_lp.to(device=logits.device, dtype=torch.float32)
        kl = (p.float() * (lp_all - ref_lp)).sum(1).mean()  # KL(pi || pi_BC), exact over the legal options
        loss_pol = loss_pol + kl_coef * kl
    stats = {
        "pg": pg.detach(), "vf": vf.detach(), "ent": ent.detach(), "kl_bc": kl.detach(),
        "clipfrac": ((ratio - 1).abs() > clip).float().mean().detach(),
        "approx_kl": ((ratio - 1) - torch.log(ratio)).mean().detach(),
        "critic_ev": explained_variance(cval, ret), "policy_ev": explained_variance(pval, ret),
    }  # fmt: skip
    return loss_pol, loss_crit, stats


def window_rows(obs: dict, legal: torch.Tensor) -> torch.Tensor:
    """Bool [B]: opponent-turn, chain-0 rows with a legal activation (the torch twin of
    rollout.target_window_mask: global[5] = own turn, global[20] = chain length, actions[..., 0] = option kind)."""
    from decklight.engine.actions import Kind

    g = obs["global"]
    act = (obs["actions"][:, : legal.shape[1], 0] == int(Kind.ACTIVATE)) & legal
    return (g[:, 5] == 0) & (g[:, 20] == 0) & act.any(1)


def ppo_loss(model, ref_lp, obs, action, old_logp, adv, ret, clip: float, vf_coef: float, kl_coef: float,
             ent_coef: float, keep: torch.Tensor | None = None,
             isw: torch.Tensor | None = None, kl_win_mult: float = 1.0) -> tuple[torch.Tensor, dict]:  # fmt: skip
    """`keep` (bool [B], advantage filtering): policy terms (pg, entropy, KL to the
    reference) are means over the kept rows only; the value loss stays over all rows.
    `isw` (float [B], --win-explore): truncated importance weight multiplying the
    advantage per row; `None` (default) is exactly today's computation.
    `kl_win_mult` (--kl-win-mult): the KL term's weight on opponent-turn chain-0 rows with a legal activation
    (`window_rows`); 1.0 (default) is exactly today's computation."""
    logits, value = model.forward_value(obs)
    lp_all = masked_log_softmax(logits)
    legal = torch.isfinite(logits)
    p = lp_all.exp() * legal
    logp = lp_all.gather(1, action[:, None]).squeeze(1)
    ratio = torch.exp(logp - old_logp)
    if isw is not None:
        adv = adv * isw.to(dtype=adv.dtype)
    pg_row = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    vf = F.mse_loss(value.float(), ret)
    ent_row = -(p * lp_all).sum(1)
    if keep is None:
        pg = pg_row.mean()
        ent = ent_row.mean()
        loss = pg + vf_coef * vf - ent_coef * ent
        kl = torch.zeros((), device=logits.device)
        if ref_lp is not None and kl_coef > 0:
            ref_lp = ref_lp.to(dtype=torch.float32) if torch.is_tensor(ref_lp) else torch.as_tensor(ref_lp)
            ref_lp = ref_lp.to(device=logits.device, dtype=torch.float32)
            kl_row = (p.float() * (lp_all - ref_lp)).sum(1)  # KL(pi || pi_BC), exact over the legal options
            if kl_win_mult != 1.0:
                kl_row = kl_row * torch.where(window_rows(obs, legal), kl_win_mult, 1.0)
            kl = kl_row.mean()
            loss = loss + kl_coef * kl
        # Detached 0-d tensors (no host sync here): callers accumulate on-device and
        # convert once per iteration. Same keys/values as the old per-minibatch .item() path.
        stats = {
            "pg": pg.detach(), "vf": vf.detach(), "ent": ent.detach(), "kl_bc": kl.detach(),
            "clipfrac": ((ratio - 1).abs() > clip).float().mean().detach(),
            "approx_kl": ((ratio - 1) - torch.log(ratio)).mean().detach(),
        }  # fmt: skip
        return loss, stats
    k = keep.to(device=logits.device, dtype=torch.bool)
    m = k.to(dtype=pg_row.dtype)
    denom = m.sum().clamp_min(1)  # 0 kept rows: policy terms are 0 (no NaN), value loss untouched
    pg = (pg_row * m).sum() / denom
    ent = (ent_row * m).sum() / denom
    loss = pg + vf_coef * vf - ent_coef * ent
    kl = torch.zeros((), device=logits.device)
    if ref_lp is not None and kl_coef > 0:
        ref_lp = ref_lp.to(dtype=torch.float32) if torch.is_tensor(ref_lp) else torch.as_tensor(ref_lp)
        ref_lp = ref_lp.to(device=logits.device, dtype=torch.float32)
        kl_row = (p.float() * (lp_all - ref_lp)).sum(1)
        if kl_win_mult != 1.0:
            kl_row = kl_row * torch.where(window_rows(obs, legal), kl_win_mult, 1.0)
        kl = (kl_row * m).sum() / denom
        loss = loss + kl_coef * kl
    stats = {
        "pg": pg.detach(), "vf": vf.detach(), "ent": ent.detach(), "kl_bc": kl.detach(),
        "clipfrac": ((((ratio - 1).abs() > clip).float() * m).sum() / denom).detach(),
        "approx_kl": ((((ratio - 1) - torch.log(ratio)) * m).sum() / denom).detach(),
        "adv_keep": m.mean().detach(),
    }  # fmt: skip
    return loss, stats
