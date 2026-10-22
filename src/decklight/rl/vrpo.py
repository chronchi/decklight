"""VRPO advantages (RL v2 step 2): Q-critic + expected-SARSA(λ) as an alternative to GAE.

PLAN §2 / Phase 5: "GAE falls short in imperfect-info self-play". VRPO trains a Q-critic by
expected-SARSA(λ) and uses advantages A = Q(s,a) − V̄(s) with V̄(s) = Σ_a π(a|s) Q(s,a),
which is PPO-compatible. `--adv gae` (default) stays bit-identical; this module only runs
on the `--adv vrpo` path.

Targets (per seat trajectory, γ = 1, reward r only at the end, same as `gae`):
    G_T = r
    G_t = V̄(s_{t+1}) + λ (G_{t+1} − Q(s_{t+1}, a_{t+1}))   for t < T
(r_t = 0 except at the last step.) Stored rollout-time Q and π are used for the targets.
Policy advantage: A_t = Q(s_t, a_t) − V̄(s_t), normalized like adv_n in train_rl.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from decklight.rl.ppo import explained_variance, gae, masked_log_softmax


def vbar(pi: np.ndarray, q: np.ndarray) -> np.ndarray:
    """V̄(s) = Σ_a π(a|s) Q(s,a) per row. Illegal options carry π = 0 so any finite Q there is ignored.

    No NaN masking here: a non-finite Q must surface as a non-finite V̄/advantage (fail loud).
    `forward_q` guarantees finite Q (illegal/padded options come out as 0)."""
    pi = np.asarray(pi, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    return (pi * q).sum(-1).astype(np.float32)


def vrpo_trajectory(q_taken: np.ndarray, vbar_vals: np.ndarray, reward: float,
                    lam: float) -> tuple[np.ndarray, np.ndarray]:
    """One finished seat trajectory. Returns (advantages, returns).

    Q_t = Q(s_t, a_t) (rollout-time), V̄_t = V̄(s_t), G_T = r,
    G_t = V̄_{t+1} + λ (G_{t+1} − Q_{t+1}) for t < T; A_t = Q_t − V̄_t.
    λ = 0 → G_t = V̄_{t+1} (t < T); λ = 1 → r + Σ_{k>t}(V̄_k − Q_k).
    """
    q_taken = np.asarray(q_taken, dtype=np.float32)
    vbar_vals = np.asarray(vbar_vals, dtype=np.float32)
    n = len(q_taken)
    assert len(vbar_vals) == n
    ret = np.zeros(n, np.float32)
    if n == 0:
        return np.zeros(0, np.float32), ret
    ret[-1] = reward
    for t in range(n - 2, -1, -1):
        ret[t] = vbar_vals[t + 1] + lam * (ret[t + 1] - q_taken[t + 1])
    adv = (q_taken - vbar_vals).astype(np.float32)
    return adv, ret


def vrpo_batch(store, finished: list, lam: float):
    """Expected-SARSA(λ) per finished seat trajectory from stored rollout-time π/Q.

    Returns (slots, adv, ret, adv_n, info) where info has `vbar_mean` and `adv_diff`
    (mean |A_vrpo − A_gae| with GAE computed from V̄ as the value, unnormalized).
    """
    slots = np.concatenate([np.array(s, np.int64) for s, _ in finished])
    assert store.pi is not None and store.q is not None, "vrpo_batch needs a vrpo SlotStore (pi/Q)"
    pi = store.pi[slots].astype(np.float32)
    q = store.q[slots].astype(np.float32)
    vb = vbar(pi, q)
    act = np.asarray(store.action[slots], np.int64)
    n_opt = q.shape[1]
    safe_act = np.clip(act, 0, n_opt - 1)
    q_taken_all = q[np.arange(len(slots)), safe_act]
    # `slots` is the concatenation of the trajectories in order: trajectory i occupies [off_i, off_i + len_i).
    offs = np.cumsum([0] + [len(s) for s, _ in finished])
    positions = [np.arange(offs[i], offs[i + 1]) for i in range(len(finished))]
    advs, rets = [], []
    for (s, rew), pos in zip(finished, positions):
        ad, rt = vrpo_trajectory(q_taken_all[pos], vb[pos], rew, lam)
        advs.append(ad)
        rets.append(rt)
    adv, ret = np.concatenate(advs), np.concatenate(rets)
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    # Reference GAE from V̄ as the value, to see the two advantage choices differ.
    gae_advs = []
    for (_, rew), pos in zip(finished, positions):
        ad, _ = gae(vb[pos], rew, lam)
        gae_advs.append(ad)
    adv_diff = float(np.mean(np.abs(adv - np.concatenate(gae_advs))))
    info = {"vbar_mean": float(vb.mean()), "adv_diff": adv_diff}
    return slots, adv, ret, adv_n, info


def ppo_loss_vrpo(model, ref_lp, obs, action, old_logp, adv, ret, clip: float, vf_coef: float,
                  kl_coef: float, ent_coef: float) -> tuple[torch.Tensor, dict]:
    """PPO with the policy net's own Q head: policy loss unchanged, critic loss = MSE(Q(s,a), G)."""
    logits, q = model.forward_q(obs)
    lp_all = masked_log_softmax(logits)
    legal = torch.isfinite(logits)
    p = lp_all.exp() * legal
    logp = lp_all.gather(1, action[:, None]).squeeze(1)
    ratio = torch.exp(logp - old_logp)
    pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    qa = q.gather(1, action[:, None]).squeeze(1)
    vf = F.mse_loss(qa.float(), ret)
    ent = -(p * lp_all).sum(1).mean()
    loss = pg + vf_coef * vf - ent_coef * ent
    kl = torch.zeros((), device=logits.device)
    if ref_lp is not None and kl_coef > 0:
        ref_lp = ref_lp.to(dtype=torch.float32) if torch.is_tensor(ref_lp) else torch.as_tensor(ref_lp)
        ref_lp = ref_lp.to(device=logits.device, dtype=torch.float32)
        kl = (p.float() * (lp_all - ref_lp)).sum(1).mean()
        loss = loss + kl_coef * kl
    stats = {
        "pg": pg.detach(), "vf": vf.detach(), "ent": ent.detach(), "kl_bc": kl.detach(),
        "clipfrac": ((ratio - 1).abs() > clip).float().mean().detach(),
        "approx_kl": ((ratio - 1) - torch.log(ratio)).mean().detach(),
        "q_ev": explained_variance(qa, ret),
    }  # fmt: skip
    return loss, stats


def ppo_loss_vrpo_priv(policy, critic, ref_lp, obs, cobs, action, old_logp, adv, ret, clip: float,
                       vf_coef: float, kl_coef: float,
                       ent_coef: float) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """PPO with a privileged Q-critic: policy loss from the policy obs, Q loss from the critic obs."""
    logits, pval = policy.forward_value(obs)
    _, cq = critic.forward_q(cobs)
    lp_all = masked_log_softmax(logits)
    legal = torch.isfinite(logits)
    p = lp_all.exp() * legal
    logp = lp_all.gather(1, action[:, None]).squeeze(1)
    ratio = torch.exp(logp - old_logp)
    pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    qa = cq.gather(1, action[:, None]).squeeze(1)
    vf = F.mse_loss(qa.float(), ret)
    ent = -(p * lp_all).sum(1).mean()
    loss_pol = pg - ent_coef * ent
    loss_crit = vf_coef * vf
    kl = torch.zeros((), device=logits.device)
    if ref_lp is not None and kl_coef > 0:
        ref_lp = ref_lp.to(dtype=torch.float32) if torch.is_tensor(ref_lp) else torch.as_tensor(ref_lp)
        ref_lp = ref_lp.to(device=logits.device, dtype=torch.float32)
        kl = (p.float() * (lp_all - ref_lp)).sum(1).mean()
        loss_pol = loss_pol + kl_coef * kl
    stats = {
        "pg": pg.detach(), "vf": vf.detach(), "ent": ent.detach(), "kl_bc": kl.detach(),
        "clipfrac": ((ratio - 1).abs() > clip).float().mean().detach(),
        "approx_kl": ((ratio - 1) - torch.log(ratio)).mean().detach(),
        "q_ev": explained_variance(qa, ret), "policy_ev": explained_variance(pval, ret),
    }  # fmt: skip
    return loss_pol, loss_crit, stats
