"""RL v0: PPO self-play from a BC checkpoint, KL-anchored to the frozen BC policy (PLAN Phase 5, first feel).

    uv run python scripts/train_rl.py --init runs/<run>/bc/best.pt --out runs/<name>/rl --hours 1.5 --device cuda

Rollouts: `--envs` C++ envs over the pool decks (holdout excluded) + the gauntlet decks. Each episode is either
self-play (both seats sampled from the learner, both trained) or, with probability `--p-bc`, learner vs the frozen BC
policy (sampled, temperature 1; only the learner seat is trained). Reward +1/-1 at the end, 0 draw, `--trunc-reward`
for both seats when a game hits `--max-decisions`. Value head new (trained alone for `--value-warmup` iterations).
Loss: clipped PPO + value MSE - entropy + `--kl-coef` * KL(pi || pi_BC) over legal options.

Progress is measured only on the fixed gauntlet every `--eval-every` iterations (and at iteration 0 / the end):
greedy learner vs greedy BC (both seats, 4 train + 4 holdout decks) and vs ygo-agent 0546 (its 31 lists in
decks/gauntlet, each vs its next 4), via the same
play_match/BCPolicy path as scripts/playtest.py. Never read the training win rates as progress.

Output (`--out`): metrics.jsonl (first record "args" carries the BC architecture so BCPolicy/playtest load the
checkpoints), iter-NNNN.pt (fp16 policy weights at each eval, no value head), last.pt (full, with optimizer).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import VecEnv
from ygo_zero.engine.duel import Engine
from ygo_zero.eval.bc_policy import _run_args
from ygo_zero.model.policy import PolicyNet
from ygo_zero.rl.evaluate import AsyncEvaluator, format_eval_line, run_eval
from ygo_zero.rl.league import League
from ygo_zero.rl.critic import critic_obs
from ygo_zero.rl.ppo import SlotStore, explained_variance, gae, gather, masked_log_softmax, ppo_loss, \
    ppo_loss_priv, to_torch
from ygo_zero.rl.rollout import Collector
from ygo_zero.rl.vrpo import ppo_loss_vrpo, ppo_loss_vrpo_priv, vrpo_batch as _vrpo_batch

ROOT = Path(__file__).resolve().parents[1]


def _effective_cores() -> int:
    """Usable CPUs: sched affinity count, capped by the cgroup quota when set."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 4
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            if period > 0:
                n = min(n, max(1, quota // period))
    except (OSError, ValueError):
        pass
    return max(1, n)


def _autocast(dev: torch.device, prec: str):
    if prec == "bf16" and dev.type == "cuda":
        return torch.autocast(dev.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_net(ckpt: Path, emb: dict, a: dict, value: bool, q: bool = False) -> PolicyNet:
    net = PolicyNet(emb, d=a.get("d", 256), layers=a.get("layers", 4), heads=a.get("heads", 4), dropout=0.0,
                    hist=int(a.get("hist", 0) or 0), known=bool(a.get("known", False)), value=value,
                    q=q)  # fmt: skip
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("model", state)
    missing, unexpected = net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()},
                                              strict=False)  # fmt: skip
    ok = all(k.startswith("value_head.") or (q and k.startswith("q_head.")) for k in missing)
    assert not unexpected and ok, (missing, unexpected)
    return net


def load_critic(ckpt: Path, emb: dict, a: dict, q: bool = False) -> PolicyNet:
    """Privileged critic: PolicyNet(known=True, value=True), init from `--critic-init`.

    Non-strict load: the checkpoint may lack the known inputs (a BC checkpoint trained
    without --known) and/or the value head (any BC checkpoint); only those may be missing.
    With `q=True` the critic also carries the VRPO Q head, which may additionally be missing
    (a BC/GAE checkpoint); it stays at its zero init then, so Q(s,.) == V(s) at start.
    """
    net = PolicyNet(emb, d=a.get("d", 256), layers=a.get("layers", 4), heads=a.get("heads", 4), dropout=0.0,
                    hist=int(a.get("hist", 0) or 0), known=True, value=True, q=q)  # fmt: skip
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("model", state)
    missing, unexpected = net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()},
                                              strict=False)  # fmt: skip
    assert not unexpected and all(k.startswith("value_head.") or k.startswith("known_emb.")
                                  or (q and k.startswith("q_head."))
                                  for k in missing), (missing, unexpected)
    if any(k.startswith("known_emb.") for k in missing):
        # Zero, not N(0,1): the critic then starts as exactly the init net, except hidden rows now carry their code.
        torch.nn.init.zeros_(net.known_emb.weight)
    return net


@torch.no_grad()
def sample(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    lp = masked_log_softmax(logits)
    p = lp.exp() * torch.isfinite(logits)
    a = torch.multinomial(p, 1).squeeze(1)
    return a.cpu().numpy().astype(np.int32), lp.gather(1, a[:, None]).squeeze(1).cpu().numpy()


def policy_state(model: PolicyNet) -> dict:
    return {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()
            if not k.startswith("value_head.") and not k.startswith("q_head.")}  # fmt: skip


class CollectTask:
    """One `col.collect(n)` in a background thread; `join` re-raises the original error, never hangs."""

    def __init__(self, col: Collector, n_samples: int) -> None:
        self.col = col
        self.n = n_samples
        self.finished = None
        self.stats = None
        self.elapsed = 0.0
        self._exc = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "CollectTask":
        self._thread.start()
        return self

    def _run(self) -> None:
        t0 = time.time()
        # Own CUDA stream, so rollout forwards can run alongside the update's kernels on the
        # default stream (collect syncs each forward to the host via .cpu() before using it).
        try:
            dev = getattr(self.col, "device", None)
            cuda = dev is not None and dev.type == "cuda"
            with torch.cuda.stream(torch.cuda.Stream(dev)) if cuda else contextlib.nullcontext():
                finished, stats = self.col.collect(self.n)
        except BaseException:
            self._exc = sys.exc_info()
            return
        self.finished, self.stats = finished, stats
        self.elapsed = time.time() - t0

    def join(self, timeout: float | None = None):
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("collect thread did not finish")
        if self._exc is not None:
            _, exc, tb = self._exc
            raise exc.with_traceback(tb)
        return self.finished, self.stats


def sync_actor(actor: PolicyNet, model: PolicyNet) -> None:
    """Copy fresh training weights into the rollout actor (call only with no collect running)."""
    actor.load_state_dict(model.state_dict())
    actor.eval()


def league_snapshot(model: PolicyNet, league: League, it: int, snap_every: int, league_size: int) -> None:
    """Every `snap_every` iterations freeze the current policy as a league opponent.

    A deepcopy (same device and precision, eval mode, gradients off); the rollout
    only calls it for logits, so the value head rides along unused. Call only when
    no collect is running (overlap mode: the collector forwards the actor copy).
    """
    if league_size <= 0 or it % snap_every != 0:
        return
    snap = copy.deepcopy(model).eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    league.add_snapshot(f"snap:{it}", snap)


def league_line(it: int, st: dict) -> str:
    """One compact per-iteration league line (games / learner wr / sampling weight)."""
    parts = [f"{name} g={d['games']} wr={d['wr']:.3f} w={d['pfsp_w']:.3f}"
             for name, d in st.get("league", {}).items()]
    return f"[league it {it}] " + " | ".join(parts)


def league_fail_checks(it: int, st: dict, league: League, snap_every: int, league_size: int, fail) -> None:
    if st.get("opp_games", 0) > 0 and st.get("opp_learner_samples", 0) == 0:
        fail(f"iteration {it}: {st['opp_games']} opponent episodes but 0 learner samples from them")
    if league_size > 0 and it >= 2 * snap_every and league.added == 0:
        fail(f"iteration {it}: snapshots enabled (--snap-every {snap_every}) but none added")


def priv_fail_checks(it: int, st: dict, priv_critic: bool, fail) -> None:
    if priv_critic and st.get("priv_rows", 0) > 0 and st.get("priv_nonzero", 0) == 0:
        fail(f"iteration {it}: priv_code all zero over {st['priv_rows']} learner rows "
             f"(the privileged feature is not reaching the critic)")


def gae_batch(store: SlotStore, finished: list, lam: float):
    """GAE per finished seat trajectory. Returns (slots, adv, ret, adv_n)."""
    slots = np.concatenate([np.array(s, np.int64) for s, _ in finished])
    advs, rets = [], []
    for s, rew in finished:
        ad, rt = gae(store.value[np.array(s)], rew, lam)
        advs.append(ad)
        rets.append(rt)
    adv, ret = np.concatenate(advs), np.concatenate(rets)
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    return slots, adv, ret, adv_n


def vrpo_targets_batch(store: SlotStore, finished: list, lam: float):
    """Expected-SARSA(λ) per finished seat trajectory (VRPO). Returns (slots, adv, ret, adv_n, info).

    `info` has `vbar_mean` and `adv_diff` (mean |A_vrpo − A_gae| with GAE from V̄ as the value).
    Raises RuntimeError on non-finite Q, V̄, advantages or returns (fail loud)."""
    slots, adv, ret, adv_n, info = _vrpo_batch(store, finished, lam)
    bad = (~np.isfinite(adv)).any() or (~np.isfinite(ret)).any()
    if bad or not np.isfinite(info["vbar_mean"]) or not np.isfinite(info["adv_diff"]):
        raise RuntimeError("non-finite Q/Vbar/advantage/return in VRPO targets")
    return slots, adv, ret, adv_n, info


def ppo_update(model: PolicyNet, upd_model, opt, vopt, store: SlotStore, slots: np.ndarray,
               adv_n: np.ndarray, ret: np.ndarray, rng: np.random.Generator, dev: torch.device,
               *, clip: float, vf_coef: float, kl_coef: float, ent_coef: float, epochs: int,
               minibatch: int, warm: bool, precision: str, max_upload_gb: float, compile: bool,
               critic: PolicyNet | None = None, copt=None, priv_critic: bool = False,
               adv: str = "gae") -> dict:
    """Upload-once + PPO epochs over one batch. Returns per-iteration mean stats (raises on non-finite).

    With `priv_critic`, the value loss trains `critic` (via `copt`) on the privileged
    obs built from the stored `priv_code`, while the policy loss (pg + KL - entropy)
    trains the policy; stats gain `critic_ev` / `policy_ev`. `--value-warmup` then
    trains the critic alone.

    With `adv="vrpo"` the critic loss is MSE of Q(s,a) to the expected-SARSA(λ) target G
    (same coef/optimizer as the value loss); the policy loss is unchanged. Warm-up trains
    the Q head only. Stats gain `q_ev` (+ `adv_diff` / `vbar_mean` are logged by the caller
    from the batch targets). `adv="gae"` (default) is bit-identical to before.
    """
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    if priv_critic and (critic is None or copt is None or "priv_code" not in store.obs):
        raise ValueError("--priv-critic 1 needs a critic, its optimizer and priv_code in the slot store")
    cobs_np = None
    if priv_critic:
        assert critic is not None
        policy_keys = [k for k in store.obs if k != "priv_code"]
        cobs_np = critic_obs({k: np.ascontiguousarray(store.obs[k][slots]) for k in policy_keys},
                             np.ascontiguousarray(store.obs["priv_code"][slots]))
    per_slot = (sum(int(np.prod(v.shape[1:], dtype=np.int64)) * np.dtype(v.dtype).itemsize
                    for v in store.obs.values()) + 8 + 4 + 4 + 64 * 2 + 4 + 4)
    upload_once = per_slot * len(slots) <= max_upload_gb * 1e9
    obs_dev = act_dev = olp_dev = ref_dev = adv_dev = ret_dev = perm_dev = None
    cobs_dev = None
    if upload_once:
        obs_dev = {k: torch.from_numpy(np.ascontiguousarray(store.obs[k][slots])).to(dev)
                   for k in store.obs}
        act_dev = torch.from_numpy(np.ascontiguousarray(store.action[slots])).to(dev)
        olp_dev = torch.from_numpy(np.ascontiguousarray(store.logp[slots])).to(dev)
        ref_dev = torch.from_numpy(np.ascontiguousarray(store.ref_lp[slots])).to(dev)
        adv_dev = torch.from_numpy(np.ascontiguousarray(adv_n)).to(dev)
        ret_dev = torch.from_numpy(np.ascontiguousarray(ret)).to(dev)
        perm_dev = torch.from_numpy(rng.permutation(len(slots))).to(dev)
        if priv_critic:
            assert cobs_np is not None
            cobs_dev = to_torch(cobs_np, dev)
    t_up = time.time()
    model.train()
    if priv_critic:
        assert critic is not None
        critic.train()
    if compile:
        upd_model.train()
    agg_sum: dict[str, torch.Tensor] = {}
    agg_n = 0
    bad_loss = False
    for _ in range(epochs):
        finite_epoch = torch.ones((), dtype=torch.bool, device=dev)
        perm = rng.permutation(len(slots))
        mb_dev = torch.from_numpy(perm).to(dev) if upload_once else None
        for k in range(0, len(perm), minibatch):
            if upload_once:
                idx = mb_dev[k : k + minibatch]
                obs = {key: v[idx] for key, v in obs_dev.items()}
                act, olp = act_dev[idx], olp_dev[idx]
                rl = ref_dev[idx] if kl_coef > 0 else None
                a_t = adv_dev[idx]
                r_t = ret_dev[idx]
                cobs_t = {key: v[idx] for key, v in cobs_dev.items()} if priv_critic else None
            else:
                mb = perm[k : k + minibatch]
                sl = slots[mb]
                obs = to_torch({key: v[sl] for key, v in store.obs.items()}, dev)
                act = torch.from_numpy(store.action[sl]).to(dev)
                olp = torch.from_numpy(store.logp[sl]).to(dev)
                rl = torch.from_numpy(store.ref_lp[sl]).to(dev) if kl_coef > 0 else None
                a_t = torch.from_numpy(adv_n[mb]).to(dev)
                r_t = torch.from_numpy(ret[mb]).to(dev)
                cobs_t = to_torch({key: v[mb] for key, v in cobs_np.items()}, dev) \
                    if priv_critic else None
            if warm:
                if adv == "vrpo" and priv_critic:
                    assert critic is not None and copt is not None
                    with _autocast(dev, precision):
                        _, cq = critic.forward_q(cobs_t)
                        qa = cq.gather(1, act[:, None]).squeeze(1)
                        loss = torch.nn.functional.mse_loss(qa.float(), r_t)
                    finite_epoch = finite_epoch & torch.isfinite(loss).detach()
                    copt.zero_grad(set_to_none=True)
                    loss.backward()
                    for n, p in critic.named_parameters():  # Q head only: trunk/value stay frozen
                        if not n.startswith("q_head.") and p.grad is not None:
                            p.grad = None
                    copt.step()
                    s = {"vf": loss.detach(), "q_ev": explained_variance(qa, r_t)}
                elif adv == "vrpo":
                    with _autocast(dev, precision):
                        _, mq = upd_model.forward_q(obs)
                        qa = mq.gather(1, act[:, None]).squeeze(1)
                        loss = torch.nn.functional.mse_loss(qa.float(), r_t)
                    finite_epoch = finite_epoch & torch.isfinite(loss).detach()
                    vopt.zero_grad(set_to_none=True)
                    loss.backward()
                    vopt.step()
                    s = {"vf": loss.detach(), "q_ev": explained_variance(qa, r_t)}
                elif priv_critic:
                    assert critic is not None and copt is not None
                    with _autocast(dev, precision):
                        with torch.no_grad():  # critic value head only: no graph through the trunk
                            _, cenc = critic._run(cobs_t)
                        cv = critic.value_head(cenc[:, 0]).squeeze(-1)
                        loss = torch.nn.functional.mse_loss(cv.float(), r_t)
                    finite_epoch = finite_epoch & torch.isfinite(loss).detach()
                    copt.zero_grad(set_to_none=True)
                    loss.backward()
                    copt.step()
                    s = {"vf": loss.detach(), "critic_ev": explained_variance(cv, r_t)}
                else:
                    with _autocast(dev, precision):
                        with torch.no_grad():  # value head only: no graph through the trunk
                            _, enc = upd_model._run(obs)
                        v = upd_model.value_head(enc[:, 0]).squeeze(-1)
                        loss = torch.nn.functional.mse_loss(v.float(), r_t)
                    finite_epoch = finite_epoch & torch.isfinite(loss).detach()
                    vopt.zero_grad(set_to_none=True)
                    loss.backward()
                    vopt.step()
                    s = {"vf": loss.detach()}
            elif priv_critic:
                assert critic is not None and copt is not None
                with _autocast(dev, precision):
                    if adv == "vrpo":
                        loss_pol, loss_crit, s = ppo_loss_vrpo_priv(upd_model, critic, rl, obs, cobs_t,
                                                                    act, olp, a_t, r_t, clip, vf_coef,
                                                                    kl_coef, ent_coef)  # fmt: skip
                    else:
                        loss_pol, loss_crit, s = ppo_loss_priv(upd_model, critic, rl, obs, cobs_t, act,
                                                               olp, a_t, r_t, clip, vf_coef, kl_coef,
                                                               ent_coef)  # fmt: skip
                finite_epoch = finite_epoch & torch.isfinite(loss_pol).detach() \
                    & torch.isfinite(loss_crit).detach()
                opt.zero_grad(set_to_none=True)
                loss_pol.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                copt.zero_grad(set_to_none=True)
                loss_crit.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                copt.step()
            else:
                with _autocast(dev, precision):
                    if adv == "vrpo":
                        loss, s = ppo_loss_vrpo(upd_model, rl, obs, act, olp, a_t, r_t, clip, vf_coef,
                                                kl_coef, ent_coef)  # fmt: skip
                    else:
                        loss, s = ppo_loss(upd_model, rl, obs, act, olp, a_t, r_t, clip, vf_coef,
                                           kl_coef, ent_coef)  # fmt: skip
                finite_epoch = finite_epoch & torch.isfinite(loss).detach()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            for key, val in s.items():
                t = val.detach()
                if key in agg_sum:
                    agg_sum[key] = agg_sum[key] + t
                else:
                    agg_sum[key] = t.clone()
            agg_n += 1
        if not bool(finite_epoch.item()):  # single host sync per epoch
            bad_loss = True
            break
    if bad_loss:
        raise RuntimeError("non-finite loss")
    agg_mean = {k: float((v / agg_n).item()) for k, v in agg_sum.items()} if agg_n else {}
    t_update = time.time() - t_up
    model.eval()
    if priv_critic:
        assert critic is not None
        critic.eval()
    return agg_mean, t_update


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="BC checkpoint (its metrics.jsonl gives the architecture)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--envs", type=int, default=512)  # open trajectories must fit --store (≈ envs × 550)
    _eff = _effective_cores()
    _eval_default = max(1, _eff // 2)
    ap.add_argument("--threads", type=int, default=max(2, _eff - _eval_default))
    ap.add_argument("--eval-mode", choices=("async", "sync"), default="async",
                    help="async: eval_ckpt.py subprocess while training continues; sync: blocking in-process")
    ap.add_argument("--eval-threads", type=int, default=_eval_default,
                    help="CPUs for the eval subprocess (async mode)")
    ap.add_argument("--recv-batch", type=int, default=None, help="async recv batch (default: envs // 4)")
    ap.add_argument("--precision", default=None, help="fp32|bf16 (default: bf16 on cuda, fp32 on cpu)")
    ap.add_argument("--max-upload-gb", type=float, default=6.0, help="device-upload-once budget per iteration")
    ap.add_argument("--max-decisions", type=int, default=1500, help="per game, both seats")
    ap.add_argument("--trunc-reward", type=float, default=-0.5)
    ap.add_argument("--p-bc", type=float, default=0.3, help="share of episodes vs the frozen BC policy (ignored with --league 1)")
    ap.add_argument("--league", type=int, choices=(0, 1), default=0,
                    help="0: today's self-play/vs-BC mix (default); 1: PFSP league (snapshots + ygo-agent)")
    ap.add_argument("--p-self", type=float, default=0.4, help="league: share of self-play episodes")
    ap.add_argument("--p-ygoagent", type=float, default=0.15,
                    help="league: absolute share of ygo-agent episodes (0 = off)")
    ap.add_argument("--snap-every", type=int, default=40, help="league: freeze a snapshot every N iterations")
    ap.add_argument("--league-size", type=int, default=6,
                    help="league: newest snapshots kept as opponents (0 = no snapshots)")
    ap.add_argument("--rollout", type=int, default=32768, help="finished samples per PPO iteration")
    ap.add_argument("--store", type=int, default=400_000, help="sample slots (open + finished trajectories)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--adv", choices=("gae", "vrpo"), default="gae",
                    help="gae: GAE advantages (default); vrpo: Q-critic + expected-SARSA(λ) advantages")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--value-warmup", type=int, default=2, help="first iterations train the value head only")
    ap.add_argument("--priv-critic", type=int, choices=(0, 1), default=0,
                    help="1: separate privileged critic (sees hidden cards); the policy stays honest")
    ap.add_argument("--critic-init", type=Path, default=None,
                    help="critic init checkpoint (default: --init)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=6, help="vs BC: seeds per ordered deck pair")
    ap.add_argument("--ga-seeds", type=int, default=1, help="vs ygo-agent: seeds per deck pair")
    ap.add_argument("--gauntlet-decks", type=Path, default=ROOT / "decks/gauntlet",
                    help="ygo-agent's own lists (in its 864-card vocabulary); also in the training pool")
    ap.add_argument("--ga-opps", type=int, default=4, help="vs ygo-agent: each gauntlet deck vs its next k decks")
    ap.add_argument("--no-ygoagent", action="store_true")
    ap.add_argument("--hours", type=float, default=1.0, help="stop (then final eval) after this much training")
    ap.add_argument("--iters", type=int, default=0, help="stop after this many iterations (0 = by --hours)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fused-adam", dest="fused_adam", action=argparse.BooleanOptionalAction, default=None,
                    help="fused AdamW kernels (default: on when --device is cuda)")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="torch.compile(model, dynamic=True) for the update forward only (rollout stays eager)")
    ap.add_argument("--overlap", action="store_true", default=False,
                    help="overlap the next rollout with the PPO update (background collect, one-iteration lag)")
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    if args.recv_batch is None:
        args.recv_batch = max(1, args.envs // 4)
    if args.precision is None:
        args.precision = "bf16" if dev.type == "cuda" else "fp32"
    if args.fused_adam is None:
        args.fused_adam = dev.type == "cuda"

    def fail(msg: str):
        line = f"ALERT train_rl: {msg}"
        print(line, flush=True)
        try:
            with open(args.out / "ALERT", "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
        raise SystemExit(3)

    def fail2(msg: str):
        """Startup usage error (exit 2): --adv vrpo without a Q-capable critic."""
        line = f"ALERT train_rl: {msg}"
        print(line, flush=True)
        try:
            with open(args.out / "ALERT", "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
        raise SystemExit(2)

    def alert(msg: str):
        """Non-fatal ALERT (training continues): q_ev collapse warning."""
        line = f"ALERT train_rl: {msg}"
        print(line, flush=True)
        try:
            with open(args.out / "ALERT", "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    qev_bad = 0

    def note_qev(it: int, agg_mean: dict, warm: bool) -> None:
        """q_ev < -1 for 5 consecutive iterations after warm-up → ALERT (no exit)."""
        nonlocal qev_bad
        if args.adv != "vrpo" or warm:
            return
        qev = agg_mean.get("q_ev")
        if qev is not None and qev < -1:
            qev_bad += 1
            if qev_bad >= 5:
                alert(f"iteration {it}: q_ev {qev:.3f} < -1 for 5 consecutive iterations")
                qev_bad = 0
        else:
            qev_bad = 0
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)

    # decks: training pool = pool - holdout + gauntlet; eval = fixed 4 train + 4 holdout sample
    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    pool = sorted((ROOT / "decks/pool").glob("*.ydk"))
    gfiles = sorted(args.gauntlet_decks.glob("*.ydk"))
    train_decks = [load_ydk(p) for p in pool if p.stem not in hold] + [load_ydk(p) for p in gfiles]
    r = random.Random(1234)
    ev_names = r.sample([p.stem for p in pool if p.stem not in hold], 4) + r.sample(sorted(hold & {p.stem for p in pool}), 4)
    ev_decks = [load_ydk(ROOT / "decks/pool" / f"{n}.ydk") for n in ev_names]
    g_decks = [load_ydk(p) for p in gfiles]

    a = _run_args(args.init)
    hist, known = int(a.get("hist", 0) or 0), bool(a.get("known", False))
    emb = dict(np.load(args.emb))
    use_q_policy = args.adv == "vrpo" and not args.priv_critic
    use_q_critic = args.adv == "vrpo" and bool(args.priv_critic)
    model = load_net(args.init, emb, a, value=True, q=use_q_policy).to(dev).eval()
    ref = load_net(args.init, emb, a, value=False).to(dev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    vparams = list(model.value_head.parameters())
    trunk = [p for n, p in model.named_parameters() if not n.startswith("value_head.")]
    _fused = {"fused": True} if args.fused_adam and dev.type == "cuda" else {}
    opt = torch.optim.AdamW([{"params": trunk}, {"params": vparams, "lr": 1e-4}], lr=args.lr,
                            weight_decay=0.0, **_fused)
    if use_q_policy:
        vopt = torch.optim.AdamW(model.q_head.parameters(), lr=3e-4, **_fused)
    else:
        vopt = torch.optim.AdamW(model.value_head.parameters(), lr=3e-4, **_fused)
    critic, copt = None, None
    if args.priv_critic:
        critic = load_critic(args.critic_init or args.init, emb, a, q=use_q_critic).to(dev).eval()
        copt = torch.optim.AdamW(critic.parameters(), lr=3e-4, **_fused)
    if args.adv == "vrpo":
        owner = critic if args.priv_critic else model
        if owner is None or not getattr(owner, "has_q", False):
            fail2("--adv vrpo needs a Q-capable critic "
                  f"({'privileged critic' if args.priv_critic else 'policy net'} built with q=True)")
    # Update forward only: rollout Collector keeps the eager `model`; the compiled
    # module shares parameters with it, so optimizer steps and saved state_dicts are unchanged.
    upd_model = torch.compile(model, dynamic=True) if args.compile else model

    log = (args.out / "metrics.jsonl").open("a")
    arch = {k: a.get(k) for k in ("d", "layers", "heads", "hist", "known")}
    log.write(json.dumps({"args": {**arch, "rl": {k: str(v) for k, v in vars(args).items()}}, "eval_decks": ev_names,
                          "train_decks": len(train_decks)}) + "\n")  # fmt: skip
    log.flush()

    engine = Engine()

    def save_critic_pt() -> None:
        """priv-critic only: fp16 critic weights next to the iter checkpoints (for search later)."""
        if args.priv_critic:
            assert critic is not None
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in critic.state_dict().items()},
                       args.out / "critic.pt")

    def save_last(it: int) -> None:
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "iter": it}
        if args.priv_critic:
            assert critic is not None and copt is not None
            ckpt["critic"] = critic.state_dict()
            ckpt["copt"] = copt.state_dict()
        torch.save(ckpt, args.out / "last.pt")

    def evaluate(it: int) -> dict:
        """Synchronous eval: save iter-NNNN.pt, run in-process, print + log the record."""
        ck = args.out / f"iter-{it:04d}.pt"
        torch.save(policy_state(model), ck)
        save_critic_pt()
        res = run_eval(ck, args.init, args.emb, ev_names, hold, args.gauntlet_decks, args.eval_seeds,
                       args.ga_seeds, args.ga_opps, args.no_ygoagent, args.device, args.threads, it=it)  # fmt: skip
        print(format_eval_line(res), flush=True)
        log.write(json.dumps(res) + "\n")
        log.flush()
        return res

    asev: AsyncEvaluator | None = None
    recorded: set[int] = set()
    eval_started: set[int] = set()

    def relay(recs: list[dict]) -> None:
        for r in recs:
            print(format_eval_line(r), flush=True)
            log.write(json.dumps(r) + "\n")
            recorded.add(r.get("iter"))
        if recs:
            log.flush()

    if args.eval_mode == "async":
        evals_path = args.out / "evals.jsonl"

        def _eval_cmd(it: int, ckpt: Path) -> list[str]:
            cmd = [sys.executable, str(ROOT / "scripts" / "eval_ckpt.py"), "--ckpt", str(ckpt),
                   "--init", str(args.init), "--iter", str(it), "--emb", str(args.emb),
                   "--eval-names", ",".join(ev_names), "--gauntlet-decks", str(args.gauntlet_decks),
                   "--eval-seeds", str(args.eval_seeds), "--ga-seeds", str(args.ga_seeds),
                   "--ga-opps", str(args.ga_opps), "--device", str(args.device),
                   "--threads", str(args.eval_threads), "--out", str(evals_path)]  # fmt: skip
            if args.no_ygoagent:
                cmd.append("--no-ygoagent")
            return cmd

        asev = AsyncEvaluator(_eval_cmd, args.out)

    evaluate(0)
    recorded.add(0)
    eval_started.add(0)

    # rollout state: async collector owns the envs + trajectory bookkeeping
    N = args.envs
    vec = VecEnv(engine, train_decks, N, args.threads, seed=args.seed)
    vec.set_max_decisions(args.max_decisions)
    store = SlotStore(args.store, gather(vec.buf, np.arange(1), hist, known, priv=bool(args.priv_critic)),
                      vrpo=(args.adv == "vrpo"))
    actor = critic_actor = None
    col_rng = rng
    if args.overlap:
        # The collector runs in a background thread while the main thread updates `model`,
        # so it forwards a private actor copy (synced after each update, never mid-collect).
        # It also gets its own RNG: np Generators are not thread-safe and the main thread
        # keeps using `rng` for minibatch permutations during the update.
        actor = copy.deepcopy(model).eval()
        if args.priv_critic:
            assert critic is not None
            critic_actor = copy.deepcopy(critic).eval()
        col_rng = np.random.default_rng(args.seed + 0x1234)
    league = None
    ygo_pol = None
    if args.league:
        from ygo_zero.eval.ygoagent import MODEL_PATH as _YGO_MODEL

        if args.p_ygoagent > 0 and args.p_self < 1 and not _YGO_MODEL.is_file():
            fail(f"--league 1 needs the ygo-agent model for --p-ygoagent {args.p_ygoagent}: "
                 f"{_YGO_MODEL} missing")
        league = League(args.p_self, args.p_ygoagent, args.league_size, col_rng)
        if args.p_ygoagent > 0 and args.p_self < 1:
            from ygo_zero.eval.ygoagent import YgoAgentPolicy

            ygo_pol = YgoAgentPolicy(seed=args.seed + 0x5EED, sample=True)
    try:
        col = Collector(vec, actor if actor is not None else model, ref, store, hist, known, dev,
                        args.p_bc, args.trunc_reward, args.recv_batch, col_rng, args.precision,
                        league, ygo_pol,
                        critic=(critic_actor if critic_actor is not None else critic),
                        priv_critic=bool(args.priv_critic), adv=args.adv)
    except ValueError as e:
        fail2(str(e))
    it, t_train = 0, time.time()
    tot = {"games": 0, "trunc": 0, "decisions": 0}
    if args.overlap:
        t0 = time.time()
        pending = col.collect(args.rollout)
        pending_roll = time.time() - t0
        if not pending[0]:
            fail("iteration 1: collected 0 samples")
        while True:
            it += 1
            t_it = time.time()
            finished, st = pending
            t_roll = pending_roll
            if league is not None:
                league_fail_checks(it, st, league, args.snap_every, args.league_size, fail)
            priv_fail_checks(it, st, bool(args.priv_critic), fail)
            # Start the next batch in the background (actor weights: one update behind
            # after the first sync); the last --iters iteration has no next batch.
            task = None
            if not (args.iters and it >= args.iters):
                task = CollectTask(col, args.rollout).start()
            vrpo_info: dict = {}
            if args.adv == "vrpo":
                try:
                    slots, adv, ret, adv_n, vrpo_info = vrpo_targets_batch(store, finished, args.lam)
                except RuntimeError:
                    fail(f"iteration {it}: non-finite Q/Vbar/advantage/return")
            else:
                slots, adv, ret, adv_n = gae_batch(store, finished, args.lam)
            if len(slots) == 0:
                fail(f"iteration {it}: collected 0 samples")
            warm = it <= args.value_warmup
            try:
                agg_mean, t_update = ppo_update(
                    model, upd_model, opt, vopt, store, slots, adv_n, ret, rng, dev,
                    clip=args.clip, vf_coef=args.vf_coef, kl_coef=args.kl_coef,
                    ent_coef=args.ent_coef, epochs=args.epochs, minibatch=args.minibatch,
                    warm=warm, precision=args.precision, max_upload_gb=args.max_upload_gb,
                    compile=args.compile, critic=critic, copt=copt,
                    priv_critic=bool(args.priv_critic), adv=args.adv)
            except RuntimeError:
                fail(f"iteration {it}: non-finite loss")
            note_qev(it, agg_mean, warm)
            wait_s = 0.0
            if task is not None:
                t_join = time.time()
                try:
                    pending = task.join()
                except BaseException as e:
                    line = f"ALERT train_rl: overlap collect failed at iteration {it}: {e!r}"
                    print(line, flush=True)
                    try:
                        with open(args.out / "ALERT", "a") as f:
                            f.write(line + "\n")
                    except OSError:
                        pass
                    raise
                wait_s = time.time() - t_join
                pending_roll = task.elapsed
                if not pending[0]:
                    fail(f"iteration {it}: collected 0 samples")
            for s, _ in finished:
                store.release(s)
            assert actor is not None
            sync_actor(actor, model)
            if critic_actor is not None:
                assert critic is not None
                sync_actor(critic_actor, critic)
            if league is not None:
                league_snapshot(model, league, it, args.snap_every, args.league_size)

            if not warm and args.kl_coef > 0 and agg_mean.get("kl_bc") is not None and agg_mean["kl_bc"] == 0.0:
                fail(f"iteration {it}: kl_bc exactly 0.0 with --kl-coef {args.kl_coef}")
            dt = time.time() - t_it
            tot["games"] += st["games"]
            tot["trunc"] += st["trunc"]
            tot["decisions"] += st["decisions"]
            rec = {
                "iter": it, "warmup": warm, "samples": len(slots), "games": st["games"],
                "trunc_rate": round(st["trunc"] / max(st["games"], 1), 3),
                "mean_game_decisions": round(float(np.mean(st["lengths"])) if st["lengths"] else 0.0, 1),
                "train_vs_bc_sampled": round(st["bc_wins"] / max(st["bc_games"], 1), 3), "bc_games": st["bc_games"],
                "adv_std": round(float(adv.std()), 3), "ret_mean": round(float(ret.mean()), 3),
                "dps": round(st["decisions"] / t_roll), "rollout_s": round(t_roll, 1), "iter_s": round(dt, 1),
                "t_env": round(st["t_env"], 1), "t_fwd": round(st["t_fwd"], 1),
                "update_s": round(t_update, 1), "sps": round(len(slots) / dt, 1),
                "store_free": store.free, "hours": round((time.time() - t_train) / 3600, 3),
                "overlap": True, "wait_s": round(wait_s, 2),
                **({"vbar_mean": round(vrpo_info["vbar_mean"], 4),
                    "adv_diff": round(vrpo_info["adv_diff"], 4)} if vrpo_info else {}),
                **({"league": st["league"], "t_ga": round(st["t_ga"], 1)} if league is not None else {}),
                **{k: round(v, 4) for k, v in agg_mean.items()},
            }  # fmt: skip
            print(json.dumps(rec), flush=True)
            if league is not None:
                print(league_line(it, st), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            out_of_time = (time.time() - t_train) / 3600 >= args.hours or (args.iters and it >= args.iters)
            if args.eval_mode == "async":
                assert asev is not None
                relay(asev.poll())
            if it % args.eval_every == 0 or out_of_time:
                save_last(it)
                if args.eval_mode == "sync":
                    evaluate(it)
                    recorded.add(it)
                    eval_started.add(it)
                else:
                    assert asev is not None
                    ck = args.out / f"iter-{it:04d}.pt"
                    torch.save(policy_state(model), ck)
                    save_critic_pt()
                    if asev.maybe_start(it, ck):
                        eval_started.add(it)
                    else:
                        print(f"[eval it {it}] skipped (previous eval running)", flush=True)
            if out_of_time:
                if task is not None:
                    # Speculatively collected one batch too many: discard it, free its slots.
                    for s, _ in pending[0]:
                        store.release(s)
                if args.eval_mode == "async":
                    assert asev is not None
                    relay(asev.poll())
                    relay(asev.wait())
                    if it not in recorded and it not in eval_started:
                        evaluate(it)  # final eval was skipped: run it blocking so it is never missing
                        recorded.add(it)
                        eval_started.add(it)
                break
    while not args.overlap:
        it += 1
        t_it = time.time()
        finished, st = col.collect(args.rollout)
        t_roll = time.time() - t_it
        if not finished:
            fail(f"iteration {it}: collected 0 samples")
        if league is not None:
            league_fail_checks(it, st, league, args.snap_every, args.league_size, fail)
        priv_fail_checks(it, st, bool(args.priv_critic), fail)

        # GAE per finished seat trajectory, then PPO epochs
        # (VRPO: expected-SARSA(λ) targets from the stored rollout-time π/Q instead)
        vrpo_info2: dict = {}
        if args.adv == "vrpo":
            try:
                slots, adv, ret, adv_n, vrpo_info2 = vrpo_targets_batch(store, finished, args.lam)
            except RuntimeError:
                fail(f"iteration {it}: non-finite Q/Vbar/advantage/return")
        else:
            slots, adv, ret, adv_n = gae_batch(store, finished, args.lam)
        if len(slots) == 0:
            fail(f"iteration {it}: collected 0 samples")
        warm = it <= args.value_warmup
        try:
            agg_mean, t_update = ppo_update(
                model, upd_model, opt, vopt, store, slots, adv_n, ret, rng, dev,
                clip=args.clip, vf_coef=args.vf_coef, kl_coef=args.kl_coef,
                ent_coef=args.ent_coef, epochs=args.epochs, minibatch=args.minibatch,
                warm=warm, precision=args.precision, max_upload_gb=args.max_upload_gb,
                compile=args.compile, critic=critic, copt=copt,
                priv_critic=bool(args.priv_critic), adv=args.adv)
        except RuntimeError:
            fail(f"iteration {it}: non-finite loss")
        note_qev(it, agg_mean, warm)
        for s, _ in finished:
            store.release(s)
        if league is not None:
            league_snapshot(model, league, it, args.snap_every, args.league_size)

        if not warm and args.kl_coef > 0 and agg_mean.get("kl_bc") is not None and agg_mean["kl_bc"] == 0.0:
            fail(f"iteration {it}: kl_bc exactly 0.0 with --kl-coef {args.kl_coef}")
        dt = time.time() - t_it
        tot["games"] += st["games"]
        tot["trunc"] += st["trunc"]
        tot["decisions"] += st["decisions"]
        rec = {
            "iter": it, "warmup": warm, "samples": len(slots), "games": st["games"],
            "trunc_rate": round(st["trunc"] / max(st["games"], 1), 3),
            "mean_game_decisions": round(float(np.mean(st["lengths"])) if st["lengths"] else 0.0, 1),
            "train_vs_bc_sampled": round(st["bc_wins"] / max(st["bc_games"], 1), 3), "bc_games": st["bc_games"],
            "adv_std": round(float(adv.std()), 3), "ret_mean": round(float(ret.mean()), 3),
            "dps": round(st["decisions"] / t_roll), "rollout_s": round(t_roll, 1), "iter_s": round(dt, 1),
            "t_env": round(st["t_env"], 1), "t_fwd": round(st["t_fwd"], 1),
                "update_s": round(t_update, 1), "sps": round(len(slots) / dt, 1),
                "store_free": store.free, "hours": round((time.time() - t_train) / 3600, 3),
                **({"vbar_mean": round(vrpo_info2["vbar_mean"], 4),
                    "adv_diff": round(vrpo_info2["adv_diff"], 4)} if vrpo_info2 else {}),
                **({"league": st["league"], "t_ga": round(st["t_ga"], 1)} if league is not None else {}),
                **{k: round(v, 4) for k, v in agg_mean.items()},
            }  # fmt: skip
        print(json.dumps(rec), flush=True)
        if league is not None:
            print(league_line(it, st), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        out_of_time = (time.time() - t_train) / 3600 >= args.hours or (args.iters and it >= args.iters)
        if args.eval_mode == "async":
            assert asev is not None
            relay(asev.poll())
        if it % args.eval_every == 0 or out_of_time:
            save_last(it)
            if args.eval_mode == "sync":
                evaluate(it)
                recorded.add(it)
                eval_started.add(it)
            else:
                assert asev is not None
                ck = args.out / f"iter-{it:04d}.pt"
                torch.save(policy_state(model), ck)
                save_critic_pt()
                if asev.maybe_start(it, ck):
                    eval_started.add(it)
                else:
                    print(f"[eval it {it}] skipped (previous eval running)", flush=True)
        if out_of_time:
            if args.eval_mode == "async":
                assert asev is not None
                relay(asev.poll())
                relay(asev.wait())
                if it not in recorded and it not in eval_started:
                    evaluate(it)  # final eval was skipped: run it blocking so it is never missing
                    recorded.add(it)
                    eval_started.add(it)
            break
    print(f"done: {it} iterations, {tot['games']} games, {tot['decisions']} decisions, "
          f"{(time.time() - t_start) / 3600:.2f} h", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
