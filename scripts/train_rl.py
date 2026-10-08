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

League (`--league 1`): PFSP mix over self / base / snapshots / ygo-agent, plus fixed
checkpoints (`--league-fixed name=path` with `--league-fixed-share`) and the scripted
greedy policy (`--league-greedy-share`). `--ga-episode-share F` makes F of episodes
gauntlet-vs-gauntlet (drawn by the VecEnv); ygo-agent only plays those (share G of
them via `--ga-ygo-share`), so every ygo-agent training game is a report-card matchup.
With F = 0 (and no fixed/greedy shares) the league draws are bit-identical to before.

Curriculum (`--curric 1`, PLAN section 0a step 3): pass-rate tracking over the DAGS
starts (src/ygo_zero/rl/curriculum.py) with weights pushed every `--curric-every`
iterations; solved/hopeless starts are dropped from the PPO batch
(`--curric-drop-solved 0` keeps them). Hinted self-distillation (`--hint-bank` +
`--hint-coef C`, banked by scripts/hint_bank.py): `C x CE(policy, banked action)`
on a minibatch of banked decisions per PPO update. Both default off (bit-identical).

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
from torch.nn.parallel import DistributedDataParallel as DDP

from ygo_zero.rl.evaluate import AsyncEvaluator, format_eval_line, run_eval
from ygo_zero.rl.league import League
from ygo_zero.rl import dist as distmod
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


def dags_fail_checks(it: int, st: dict, dags_p: float, vec, fail) -> None:
    """DAGS fail-loud: p > 0 must produce DAGS episodes, and prefixes must mostly replay."""
    if dags_p > 0 and st.get("dags_games", 0) == 0:
        fail(f"iteration {it}: --dags-p {dags_p} but 0 DAGS episodes finished")
    bad, used = int(vec.dags_bad), int(vec.dags_used) + int(vec.dags_bad)
    if used > 0 and bad / used > 0.2:
        fail(f"iteration {it}: DAGS prefix failure rate {bad}/{used} > 0.2")


def ga_fail_checks(it: int, st: dict, learner_gauntlet: int, ga_share: float, fail) -> None:
    """Overfitting-rule fail-loud: the learner never pilots gauntlet decks."""
    if int(learner_gauntlet) != 0:
        return
    if int(st.get("learner_ga", 0)) > 0:
        fail(f"iteration {it}: {st['learner_ga']} learner samples on gauntlet-deck seats "
             f"(overfitting rule: --learner-gauntlet 0)")
    if ga_share > 0 and it >= 2 and int(st.get("ga_games", 0)) == 0:
        fail(f"iteration {it}: --ga-episode-share {ga_share} but 0 gauntlet episodes finished")


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


def _fwd_of(m):
    """Custom forwards (`forward_value`/`forward_q`/`_run`) bypass DDP's `forward` hook
    (DDP has no such attributes), so unwrap to the underlying module for the calls."""
    return m.module if isinstance(m, DDP) else m


def ppo_update(model: PolicyNet, upd_model, opt, vopt, store: SlotStore, slots: np.ndarray,
               adv_n: np.ndarray, ret: np.ndarray, rng: np.random.Generator, dev: torch.device,
               *, clip: float, vf_coef: float, kl_coef: float, ent_coef: float, epochs: int,
               minibatch: int, warm: bool, precision: str, max_upload_gb: float, compile: bool,
                critic: PolicyNet | None = None, copt=None, priv_critic: bool = False,
                adv: str = "gae", upd_critic=None, hint: dict | None = None) -> dict:
    """Upload-once + PPO epochs over one batch. Returns per-iteration mean stats (raises on non-finite).

    With `priv_critic`, the value loss trains `critic` (via `copt`) on the privileged
    obs built from the stored `priv_code`, while the policy loss (pg + KL - entropy)
    trains the policy; stats gain `critic_ev` / `policy_ev`. `--value-warmup` then
    trains the critic alone.

    With `adv="vrpo"` the critic loss is MSE of Q(s,a) to the expected-SARSA(λ) target G
    (same coef/optimizer as the value loss); the policy loss is unchanged. Warm-up trains
    the Q head only. Stats gain `q_ev` (+ `adv_diff` / `vbar_mean` are logged by the caller
    from the batch targets). `adv="gae"` (default) is bit-identical to before.

    With `hint` ({"obs": numpy obs dict, "act": int32 actions, "coef": C, "batch": B};
    `--hint-bank`), after the PPO epochs one extra policy step minimizes
    `C x CE(policy(obs), banked action)` on a minibatch of B banked decisions
    (hinted self-distillation: the bank holds successes found with the search's
    help; there is no hint-conditioned policy, so this is fine-tuning on hinted
    successes). Skipped during value warm-up. Stats gain `hint_ce` / `hint_acc`.
    `hint=None` (default) is bit-identical to before.

    Under DDP (process group initialized) every rank runs the same number of optimizer
    steps per epoch: the MIN local minibatch count is all-reduced and each rank drops its
    surplus tail minibatches, so no rank blocks on a collective the others have left.
    Single-process (no process group) is bit-identical to before.
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
        n_local = (len(perm) + minibatch - 1) // minibatch if len(perm) else 0
        n_common = distmod.min_int(n_local, dev) if distmod.is_dist() else n_local
        for k in range(0, min(len(perm), n_common * minibatch), minibatch):
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
            fwd = _fwd_of(upd_model)
            cfwd = _fwd_of(upd_critic if upd_critic is not None else critic)
            if warm:
                if adv == "vrpo" and priv_critic:
                    assert critic is not None and copt is not None
                    with _autocast(dev, precision):
                        _, cq = cfwd.forward_q(cobs_t)
                        qa = cq.gather(1, act[:, None]).squeeze(1)
                        loss = torch.nn.functional.mse_loss(qa.float(), r_t)
                    finite_epoch = finite_epoch & torch.isfinite(loss).detach()
                    copt.zero_grad(set_to_none=True)
                    loss.backward()
                    for n, p in critic.named_parameters():  # Q head only: trunk/value stay frozen
                        if not n.startswith("q_head.") and p.grad is not None:
                            p.grad = None
                    distmod.average_grads_(critic)
                    copt.step()
                    s = {"vf": loss.detach(), "q_ev": explained_variance(qa, r_t)}
                elif adv == "vrpo":
                    with _autocast(dev, precision):
                        _, mq = fwd.forward_q(obs)
                        qa = mq.gather(1, act[:, None]).squeeze(1)
                        loss = torch.nn.functional.mse_loss(qa.float(), r_t)
                    finite_epoch = finite_epoch & torch.isfinite(loss).detach()
                    vopt.zero_grad(set_to_none=True)
                    loss.backward()
                    distmod.average_grads_(model)
                    vopt.step()
                    s = {"vf": loss.detach(), "q_ev": explained_variance(qa, r_t)}
                elif priv_critic:
                    assert critic is not None and copt is not None
                    with _autocast(dev, precision):
                        with torch.no_grad():  # critic value head only: no graph through the trunk
                            _, cenc = cfwd._run(cobs_t)
                        cv = cfwd.value_head(cenc[:, 0]).squeeze(-1)
                        loss = torch.nn.functional.mse_loss(cv.float(), r_t)
                    finite_epoch = finite_epoch & torch.isfinite(loss).detach()
                    copt.zero_grad(set_to_none=True)
                    loss.backward()
                    distmod.average_grads_(critic)
                    copt.step()
                    s = {"vf": loss.detach(), "critic_ev": explained_variance(cv, r_t)}
                else:
                    with _autocast(dev, precision):
                        with torch.no_grad():  # value head only: no graph through the trunk
                            _, enc = fwd._run(obs)
                        v = fwd.value_head(enc[:, 0]).squeeze(-1)
                        loss = torch.nn.functional.mse_loss(v.float(), r_t)
                    finite_epoch = finite_epoch & torch.isfinite(loss).detach()
                    vopt.zero_grad(set_to_none=True)
                    loss.backward()
                    distmod.average_grads_(model)
                    vopt.step()
                    s = {"vf": loss.detach()}
            elif priv_critic:
                assert critic is not None and copt is not None
                with _autocast(dev, precision):
                    if adv == "vrpo":
                        loss_pol, loss_crit, s = ppo_loss_vrpo_priv(fwd, cfwd, rl, obs, cobs_t,
                                                                    act, olp, a_t, r_t, clip, vf_coef,
                                                                    kl_coef, ent_coef)  # fmt: skip
                    else:
                        loss_pol, loss_crit, s = ppo_loss_priv(fwd, cfwd, rl, obs, cobs_t, act,
                                                               olp, a_t, r_t, clip, vf_coef, kl_coef,
                                                               ent_coef)  # fmt: skip
                finite_epoch = finite_epoch & torch.isfinite(loss_pol).detach() \
                    & torch.isfinite(loss_crit).detach()
                opt.zero_grad(set_to_none=True)
                loss_pol.backward()
                distmod.average_grads_(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                copt.zero_grad(set_to_none=True)
                loss_crit.backward()
                distmod.average_grads_(critic)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                copt.step()
            else:
                with _autocast(dev, precision):
                    if adv == "vrpo":
                        loss, s = ppo_loss_vrpo(fwd, rl, obs, act, olp, a_t, r_t, clip, vf_coef,
                                                kl_coef, ent_coef)  # fmt: skip
                    else:
                        loss, s = ppo_loss(fwd, rl, obs, act, olp, a_t, r_t, clip, vf_coef,
                                           kl_coef, ent_coef)  # fmt: skip
                finite_epoch = finite_epoch & torch.isfinite(loss).detach()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                distmod.average_grads_(model)
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
    if hint is not None and not warm and float(hint.get("coef", 0.0)) > 0.0:
        # One hint-distillation step per PPO update (not per minibatch, so the coef
        # means the same regardless of --epochs/--minibatch). Raises on non-finite.
        hobs_np, hact_np = hint["obs"], np.ascontiguousarray(hint["act"])
        hn_all = len(hact_np)
        hn = min(int(hint.get("batch", 256)), hn_all)
        hidx = rng.choice(hn_all, hn, replace=False)
        hobs = to_torch({k: np.ascontiguousarray(v[hidx]) for k, v in hobs_np.items()}, dev)
        hact = torch.from_numpy(np.ascontiguousarray(hact_np[hidx])).to(dev)
        fwd_h = _fwd_of(upd_model)
        model.train()
        with _autocast(dev, precision):
            hlogits = fwd_h(hobs)
            hlp = masked_log_softmax(hlogits)
            hce = -hlp.gather(1, hact[:, None]).squeeze(1).mean()
            hacc = (hlogits.argmax(1) == hact).float().mean()
        if not (bool(torch.isfinite(hce).item()) and bool(torch.isfinite(hacc).item())):
            raise RuntimeError("non-finite hint loss")
        opt.zero_grad(set_to_none=True)
        (float(hint["coef"]) * hce).backward()
        distmod.average_grads_(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for key, val in (("hint_ce", hce.detach()), ("hint_acc", hacc.detach())):
            t = val.detach()
            if key in agg_sum:
                agg_sum[key] = agg_sum[key] + t
            else:
                agg_sum[key] = t.clone()
        agg_n += 1
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
    ap.add_argument("--dags-starts", type=Path, default=None,
                    help="DAGS starts file (scripts/build_dags_starts.py output); unset = no mid-game starts")
    ap.add_argument("--dags-p", type=float, default=0.0,
                    help="share of episodes starting mid-game from --dags-starts (0 = off, bit-identical)")
    ap.add_argument("--curric", type=int, choices=(0, 1), default=0,
                    help="1: pass-rate curriculum over the DAGS starts (Kolibri 3.2.2.5; "
                    "0 = uniform starts, bit-identical)")
    ap.add_argument("--curric-every", type=int, default=5,
                    help="push fresh start weights to the VecEnv every N iterations")
    ap.add_argument("--curric-eps", type=float, default=0.02,
                    help="floor weight per start (exploration + solved/hopeless share)")
    ap.add_argument("--curric-min-n", type=int, default=4,
                    help="episodes per start before it can count as solved/hopeless")
    ap.add_argument("--curric-drop-solved", type=int, choices=(0, 1), default=1,
                    help="1: drop zero-variance groups (starts solved K/K or 0/K) from the PPO batch")
    ap.add_argument("--hint-bank", type=Path, default=None,
                    help="hinted self-distillation bank (scripts/hint_bank.py output); unset = off")
    ap.add_argument("--hint-coef", type=float, default=0.0,
                    help="auxiliary loss coef C: C x CE(policy(obs), banked action) per PPO update (0 = off)")
    ap.add_argument("--hint-batch", type=int, default=256,
                    help="banked decisions per PPO update's hint step")
    ap.add_argument("--hint-start-bonus", type=float, default=0.0,
                    help="extra curriculum weight for starts present in the hint bank (0 = off, needs --curric 1)")
    ap.add_argument("--league", type=int, choices=(0, 1), default=0,
                    help="0: today's self-play/vs-BC mix (default); 1: PFSP league (snapshots + ygo-agent)")
    ap.add_argument("--p-self", type=float, default=0.4, help="league: share of self-play episodes")
    ap.add_argument("--p-ygoagent", type=float, default=0.15,
                    help="league: absolute share of ygo-agent episodes (0 = off; inert with --ga-episode-share > 0)")
    ap.add_argument("--ga-episode-share", type=float, default=0.0,
                    help="deck-aware league: share of episodes drawn gauntlet-vs-gauntlet "
                    "(0 = off, bit-identical); ygo-agent only plays those, via --ga-ygo-share")
    ap.add_argument("--ga-ygo-share", type=float, default=0.5,
                    help="within gauntlet-vs-gauntlet episodes, share played vs ygo-agent "
                    "(the rest use the league draw without YGO)")
    ap.add_argument("--learner-gauntlet", type=int, choices=(0, 1), default=0,
                    help="0: overfitting rule (default) - the learner never pilots gauntlet decks: "
                    "ordinary episodes draw from the train pool only and gauntlet episodes draw "
                    "one pool + one gauntlet deck with the learner on the pool seat; "
                    "1: today's behaviour (learner pilots gauntlet decks too), for reproducing old runs")
    ap.add_argument("--league-fixed", action="append", default=[], metavar="name=path",
                    help="extra fixed league opponent from a checkpoint (repeatable, e.g. ft-swv=path/best.pt)")
    ap.add_argument("--league-fixed-share", type=float, default=0.0,
                    help="league: absolute episode share for the --league-fixed pool (split inside by PFSP)")
    ap.add_argument("--league-greedy-share", type=float, default=0.0,
                    help="league: absolute episode share vs the scripted greedy policy (0 = off)")
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
    ap.add_argument("--ga-workers", type=int, default=8,
                    help="league ygo-agent: TFLite interpreter pool size (same moves as 1, ~3x faster)")
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
    _dev_type = torch.device(args.device).type
    dist_rank, dist_world = distmod.init_from_env(_dev_type)
    is_dist = dist_world > 1
    local_rank = int(os.environ.get("LOCAL_RANK", dist_rank)) if is_dist else 0
    seed_rank = args.seed + 1000 * dist_rank
    torch.manual_seed(seed_rank)
    rng = np.random.default_rng(seed_rank)
    dev = torch.device(args.device)
    if is_dist and dev.type == "cuda":
        dev = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(dev)
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
        if is_dist and dist_rank != 0:
            return
        line = f"ALERT train_rl: {msg}"
        print(line, flush=True)
        try:
            with open(args.out / "ALERT", "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _any_true(local: bool) -> bool:
        """Collective OR across ranks (single-process: the local value)."""
        if not is_dist:
            return local
        t = torch.tensor([1 if local else 0], dtype=torch.long, device=dev)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        return bool(t.item())

    def dfail_if_any(local_bad: bool, msg_fn) -> None:
        """Fail when ANY rank's local condition holds: ALERT from rank 0, all ranks exit 3.

        All ranks must call this together (same point in the iteration) so no rank
        blocks on a collective the others have left; torchrun then tears down the job.
        """
        bad = _any_true(local_bad)
        if bad:
            if dist_rank == 0:
                fail(msg_fn())
            else:
                raise SystemExit(3)

    def check_params_match(tag: str) -> None:
        """Iteration-1 + every-eval param checksum: ALERT + exit when ranks differ."""
        if not is_dist:
            return
        ok = distmod.checksum_ok(model, dev)
        dfail_if_any(not ok, lambda: f"{tag}: policy parameters differ across ranks")

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
    # Fail fast on incompatible curriculum/hint flags (before evaluate(0) runs any games).
    if args.curric and not args.dags_p > 0:
        fail(f"--curric 1 without --dags-p > 0 (positions are DAGS starts)")
    if (args.hint_coef > 0 or args.hint_start_bonus > 0) and args.hint_bank is None:
        fail("--hint-coef/--hint-start-bonus need --hint-bank (no bank to distil)")
    if args.hint_start_bonus > 0 and not args.curric:
        fail("--hint-start-bonus > 0 needs --curric 1")

    # decks: training pool = pool - holdout (+ gauntlet slice for ga episodes only
    # with --learner-gauntlet 0, the overfitting rule); eval = fixed 4 train + 4 holdout sample
    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    pool = sorted((ROOT / "decks/pool").glob("*.ydk"))
    gfiles = sorted(args.gauntlet_decks.glob("*.ydk"))
    pool_train = [load_ydk(p) for p in pool if p.stem not in hold]
    train_decks = pool_train + [load_ydk(p) for p in gfiles]
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
    # Under DDP the policy (and privileged critic) are wrapped for gradient averaging;
    # forwards for the update go through the DDP wrapper, the rollout keeps the eager
    # `model`. find_unused_parameters=True: the warm-up value/Q-only steps backprop
    # only the head (trunk computed under no_grad), and with --priv-critic the
    # policy's own value head never gets a gradient (ppo_loss_priv trains the
    # critic's head instead) — both leave parameters unused in some backward.
    # --compile is skipped under DDP (eager forwards); single-process is unchanged.
    if is_dist:
        ddp_kwargs = {"find_unused_parameters": True}
        if dev.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
        model_ddp = DDP(model, **ddp_kwargs)
        critic_ddp = DDP(critic, **ddp_kwargs) if critic is not None else None
        upd_model = model_ddp
        upd_critic = critic_ddp
        if args.compile and dist_rank == 0:
            print("[ddp] --compile skipped under DDP (eager forwards)", flush=True)
    else:
        upd_model = torch.compile(model, dynamic=True) if args.compile else model
        upd_critic = critic

    log = (args.out / "metrics.jsonl").open("a") if dist_rank == 0 else None
    if dist_rank == 0:
        assert log is not None
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

    def save_curriculum() -> None:
        """Write the tracker next to the checkpoints (resumable, auditable)."""
        if cur is not None and dist_rank == 0:
            cur.save(args.out / "curriculum.json")

    def evaluate(it: int) -> dict:
        """Synchronous eval: save iter-NNNN.pt, run in-process, print + log the record."""
        ck = args.out / f"iter-{it:04d}.pt"
        torch.save(policy_state(model), ck)
        save_critic_pt()
        save_curriculum()
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

    # ---- DDP (multi-GPU) per-iteration sync ---------------------------------
    def _ddp_sum(vals: list[float]) -> list[float]:
        """All-reduce SUM of a list of floats (one collective; no-op single-process)."""
        if not is_dist:
            return list(vals)
        t = torch.tensor(vals, dtype=torch.float64, device=dev)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return [float(x) for x in t.tolist()]

    def _ddp_league_sync(st: dict) -> None:
        """Average league EMAs across ranks, then sum per-opponent games/wins in place.

        Must run on every rank (collectives); only rank 0 logs the resulting global `st`.
        """
        if not is_dist or league is None:
            return
        assert league is not None
        distmod.sync_league_ema(league)
        names = sorted(st.get("league", {}))
        if not names:
            return
        games = [float(st["league"][n]["games"]) for n in names]
        wins = [float(st["league"][n]["wr"]) * float(st["league"][n]["games"]) for n in names]
        tot = _ddp_sum(games + wins)
        probs = league.sampling_probs()
        for i, n in enumerate(names):
            g, w = tot[i], tot[len(names) + i]
            st["league"][n] = {"games": int(g), "wr": round(w / max(g, 1), 3),
                               "pfsp_w": round(probs.get(n, 0.0), 4)}

    def _ddp_checks(it: int, st: dict) -> None:
        """Fail-loud checks across ranks: any rank's local trigger fails the job.

        Local messages print on their own rank (torchrun interleaves all ranks' stdout
        into train.txt); the ALERT comes from rank 0, then every rank exits non-zero.
        Single-process calls the checks with `fail` directly (bit-identical).
        """
        if not is_dist:
            if league is not None:
                league_fail_checks(it, st, league, args.snap_every, args.league_size, fail)
            priv_fail_checks(it, st, bool(args.priv_critic), fail)
            dags_fail_checks(it, st, args.dags_p, vec, fail)
            ga_fail_checks(it, st, args.learner_gauntlet, args.ga_episode_share, fail)
            return
        msgs: list[str] = []
        if league is not None:
            league_fail_checks(it, st, league, args.snap_every, args.league_size, msgs.append)
        priv_fail_checks(it, st, bool(args.priv_critic), msgs.append)
        dags_fail_checks(it, st, args.dags_p, vec, msgs.append)
        ga_fail_checks(it, st, args.learner_gauntlet, args.ga_episode_share, msgs.append)
        for m in msgs:
            print(f"ALERT train_rl: {m} (rank {dist_rank})", flush=True)
        dfail_if_any(bool(msgs),
                     lambda: f"iteration {it}: fail check triggered on some rank (see logs)")

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

    # Curriculum tracker / hint bank are set up with the VecEnv below; pre-initialized so
    # evaluate(0) (which checkpoints the tracker) can run before that.
    cur = None
    hint = None
    starts: list = []
    if dist_rank == 0:
        evaluate(0)
        recorded.add(0)
        eval_started.add(0)
    if is_dist:
        distmod.barrier()

    # rollout state: async collector owns the envs + trajectory bookkeeping
    # (--envs/--rollout/--store/--threads are per rank; seeds offset by rank).
    N = args.envs
    vec = VecEnv(engine, train_decks, N, args.threads, seed=seed_rank)
    vec.set_max_decisions(args.max_decisions)
    if args.dags_starts is not None or args.dags_p > 0:
        if args.dags_starts is None:
            fail(f"--dags-p {args.dags_p} without --dags-starts")
        from ygo_zero.rl.dags import load_starts

        try:
            starts = load_starts(args.dags_starts)
        except (OSError, ValueError) as e:
            fail(f"cannot load --dags-starts {args.dags_starts}: {e}")
        if int(args.learner_gauntlet) == 0:
            # overfitting rule: DAGS episodes are not tracked by deck index (learner_ga cannot see them), so a start
            # on a gauntlet deck would train the learner on it silently. data/cache/dags/starts.jsonl (2026-10-07)
            # is 95% gauntlet-vs-gauntlet (ygo-agent teacher games); use starts-train.jsonl (NOTEBOOK 2026-10-08).
            g_mains = {tuple(sorted(d.main)) for d in g_decks}
            n_bad = sum(1 for l in Path(args.dags_starts).read_text().splitlines() if l.strip()
                        and any(tuple(sorted(json.loads(l)[k])) in g_mains for k in ("main0", "main1")))
            if n_bad:
                fail(f"--dags-starts {args.dags_starts}: {n_bad} starts use a gauntlet deck "
                     f"(overfitting rule, --learner-gauntlet 0)")
        vec.set_starts(starts, args.dags_p, seed_rank + 0xDA65)
        if dist_rank == 0:
            print(f"[dags] {len(starts)} starts p={args.dags_p}", flush=True)
    # Hint bank (offline successes found with the search's help): replay now (engine
    # only), so a banked action that is illegal at replay fails the job here, not mid-run.
    # (Flag-compatibility failures already raised before evaluate(0); only loads remain.)
    if args.hint_bank is not None:
        from ygo_zero.rl.hints import load_bank
        from ygo_zero.rl.ppo import model_keys

        try:
            bank = load_bank(args.hint_bank, engine, hist, known, tuple(model_keys(hist, known)))
        except ValueError as e:
            fail(f"bad hint bank {args.hint_bank}: {e}")
        hint = {"obs": bank["obs"], "act": bank["act"], "coef": float(args.hint_coef),
                "batch": int(args.hint_batch), "rows": bank["rows"]}
        if dist_rank == 0:
            print(f"[hint] {bank['rows']} games, {bank['n_decisions']} decisions "
                  f"from {args.hint_bank} (coef {args.hint_coef})", flush=True)
    # Pass-rate curriculum over the DAGS starts (Kolibri 3.2.2.5): weights are pushed
    # to the VecEnv every --curric-every iterations (no push = uniform = bit-identical).
    cur = None
    if args.curric:
        if "start" not in vec.buf:
            fail("--curric 1 needs buf['start'] (rebuild the C++ env)")
        from ygo_zero.rl.curriculum import PassRate

        bonus = {}
        if args.hint_start_bonus > 0:
            assert hint is not None  # flag-compatibility checked before evaluate(0)
            bonus = {s: float(args.hint_start_bonus) for s in bank["start_ids"]
                     if 0 <= s < len(starts)}
        # slots are (start, seat) pairs: slot = 2*start + seat (rollout.py fin_starts)
        bonus = {2 * s + seat: b for s, b in bonus.items() for seat in (0, 1)}
        cur = PassRate(2 * len(starts), eps=args.curric_eps, min_n=args.curric_min_n,
                       hint_bonus=bonus)
        if dist_rank == 0:
            print(f"[curric] {len(starts)} starts every={args.curric_every} eps={args.curric_eps} "
                  f"min_n={args.curric_min_n} drop_solved={args.curric_drop_solved} "
                  f"hint_bonus_starts={len(bonus)}", flush=True)
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
        col_rng = np.random.default_rng(seed_rank + 0x1234)
    ga_start = len(pool_train)
    ga_mixed = int(args.learner_gauntlet) == 0
    if ga_mixed:
        try:
            vec.set_pool_count(len(pool_train))
        except Exception as e:  # noqa: BLE001 - C++ range errors must fail the job here
            fail(f"bad train pool (pool {len(pool_train)} of {len(train_decks)} decks): {e}")
    try:
        vec.set_ga_share(args.ga_episode_share, ga_start, len(g_decks), ga_mixed)
    except Exception as e:  # noqa: BLE001 - C++ range/share errors must fail the job here
        fail(f"bad gauntlet pool (--ga-episode-share {args.ga_episode_share}, "
             f"ga slice [{ga_start}, {ga_start + len(g_decks)})): {e}")
    if dist_rank == 0:
        print(f"[decks] pool={len(pool_train)} gauntlet slice [{ga_start}, "
              f"{ga_start + len(g_decks)}) learner_gauntlet={args.learner_gauntlet} "
              f"ga_mixed={int(ga_mixed)}", flush=True)
    if args.ga_episode_share > 0:
        if not args.league:
            fail("--ga-episode-share > 0 needs --league 1 (deck-aware draws live in the league path)")
        if not g_decks:
            fail("--ga-episode-share > 0 needs a non-empty --gauntlet-decks dir")
        if args.p_ygoagent > 0:
            print(f"[league] --ga-episode-share {args.ga_episode_share}: --p-ygoagent "
                  f"{args.p_ygoagent} is inert (ygo-agent share is "
                  f"--ga-episode-share * --ga-ygo-share = "
                  f"{args.ga_episode_share * args.ga_ygo_share:.3f})", flush=True)
    league = None
    ygo_pol = None
    greedy_pol = None
    fixed_nets: dict[str, PolicyNet] = {}
    if args.league:
        from ygo_zero.eval.ygoagent import MODEL_PATH as _YGO_MODEL

        need_ygo = (args.p_ygoagent > 0 and args.p_self < 1 and args.ga_episode_share <= 0) or \
            (args.ga_episode_share > 0 and args.ga_ygo_share > 0)
        if need_ygo and not _YGO_MODEL.is_file():
            fail(f"--league 1 needs the ygo-agent model ({_YGO_MODEL} missing)")
        for spec in args.league_fixed or []:
            if "=" not in spec:
                fail(f"--league-fixed {spec!r}: want name=path")
            name, spath = spec.split("=", 1)
            name, spath = name.strip(), Path(spath)
            if not name:
                fail(f"--league-fixed {spec!r}: empty name")
            if name in fixed_nets:
                fail(f"--league-fixed {name!r} given twice")
            if not spath.is_file():
                fail(f"--league-fixed {name!r}: {spath} missing")
            try:
                fnet = load_net(spath, emb, a, value=False).eval()
            except Exception as e:  # noqa: BLE001 - architecture mismatch must fail the job here
                fail(f"--league-fixed {name!r}: cannot load {spath} with the --init architecture: {e}")
            for p in fnet.parameters():
                p.requires_grad_(False)
            fixed_nets[name] = fnet
        if args.league_greedy_share > 0:
            from ygo_zero.eval.policies import GreedyPolicy

            greedy_pol = GreedyPolicy(seed_rank + 0x617771)
        try:
            league = League(args.p_self, args.p_ygoagent, args.league_size, col_rng,
                            fixed=fixed_nets, p_fixed=args.league_fixed_share,
                            p_greedy=args.league_greedy_share, ga_ygo_share=args.ga_ygo_share)
        except ValueError as e:
            fail2(str(e))
        if args.p_ygoagent > 0 and args.p_self < 1 and args.ga_episode_share <= 0:
            from ygo_zero.eval.ygoagent import YgoAgentPolicy

            ygo_pol = YgoAgentPolicy(seed=seed_rank + 0x5EED, sample=True, workers=args.ga_workers)
        if args.ga_episode_share > 0 and args.ga_ygo_share > 0:
            from ygo_zero.eval.ygoagent import YgoAgentPolicy

            ygo_pol = YgoAgentPolicy(seed=seed_rank + 0x5EED, sample=True, workers=args.ga_workers)
        if dist_rank == 0:
            print(f"[league] gauntlet slice [{ga_start}, {ga_start + len(g_decks)}) "
                  f"ga_share={args.ga_episode_share} ga_ygo={args.ga_ygo_share} "
                  f"fixed={sorted(fixed_nets)}", flush=True)
    try:
        col = Collector(vec, actor if actor is not None else model, ref, store, hist, known, dev,
                        args.p_bc, args.trunc_reward, args.recv_batch, col_rng, args.precision,
                        league, ygo_pol,
                        critic=(critic_actor if critic_actor is not None else critic),
                        priv_critic=bool(args.priv_critic), adv=args.adv,
                        greedy=greedy_pol, ga_share=args.ga_episode_share,
                        ga_range=(ga_start, len(g_decks)), ga_mixed=ga_mixed, fail=fail)
    except ValueError as e:
        fail2(str(e))
    curric_pushed = False

    def curric_record(finished, st: dict, it: int) -> None:
        """Feed finished learner episodes into the PassRate tracker (fail on bad indices)."""
        if cur is None:
            return
        assert cur is not None
        fin_starts = st.get("fin_starts", [])
        if len(fin_starts) != len(finished):
            fail(f"iteration {it}: {len(finished)} finished trajectories but "
                 f"{len(fin_starts)} start indices")
        for (_, rew), fst in zip(finished, fin_starts, strict=True):
            fsti = int(fst)
            if fsti < -1 or fsti >= cur.n_starts:
                fail(f"iteration {it}: buf['start'] {fsti} out of range "
                     f"[-1, {cur.n_starts})")
            if fsti >= 0:
                cur.record(fsti, 1.0 if rew > 0 else (0.5 if rew == 0 else 0.0))

    def curric_push(it: int, st: dict) -> None:
        """Push fresh start weights every --curric-every iterations (PROGRESS line)."""
        nonlocal curric_pushed
        if cur is None or it % args.curric_every != 0:
            return
        w = cur.start_weights()
        if (not w or any(not np.isfinite(x) for x in w) or not sum(w) > 0.0
                or any(x < 0.0 for x in w)):
            fail(f"iteration {it}: curriculum weights all zero or non-finite")
        try:
            vec.set_start_weights(w)
        except ValueError as e:
            fail(f"iteration {it}: cannot push curriculum weights: {e}")
        curric_pushed = True
        if dist_rank == 0:
            s = cur.stats()
            print(f"[curriculum it {it}] seen {s['seen']}/{s['starts']} "
                  f"trainable {s['trainable']} solved {s['solved']} hopeless {s['hopeless']} "
                  f"ess {s['ess_share']}", flush=True)

    def curric_check_push_seen(it: int, st: dict) -> None:
        """After the first push, DAGS episodes must carry a start index (fail loud)."""
        if cur is None or not curric_pushed:
            return
        n_dags = sum(1 for f in st.get("fin_starts", []) if int(f) >= 0)
        dfail_if_any(n_dags == 0,
                     lambda: f"iteration {it}: 0 DAGS episodes with a start index >= 0 "
                             f"after the first curriculum push")

    def curric_filter(finished, fin_starts, slots: np.ndarray, adv: np.ndarray, ret: np.ndarray,
                      adv_n: np.ndarray, it: int):
        """Drop zero-variance groups (solved K/K or hopeless 0/K starts) from the batch."""
        if cur is None or not args.curric_drop_solved:
            return slots, adv, ret, adv_n, 0
        assert cur is not None
        drop = cur.solved_mask([int(f) for f in fin_starts])
        if not any(drop):
            return slots, adv, ret, adv_n, 0
        keep = np.zeros(len(slots), bool)
        pos = 0
        for (s, _), d in zip(finished, drop, strict=True):
            n = len(s)
            if not d:
                keep[pos:pos + n] = True
            pos += n
        assert pos == len(slots), (pos, len(slots))
        n_drop = len(slots) - int(keep.sum())
        if n_drop >= len(slots):
            fail(f"iteration {it}: curriculum dropped the whole PPO batch "
                 f"({n_drop}/{len(slots)} samples from solved starts)")
        return slots[keep], adv[keep], ret[keep], adv_n[keep], n_drop

    it, t_train = 0, time.time()
    tot = {"games": 0, "trunc": 0, "decisions": 0}
    if args.overlap:
        t0 = time.time()
        pending = col.collect(args.rollout)
        pending_roll = time.time() - t0
        dfail_if_any(not pending[0], lambda: "iteration 1: collected 0 samples")
        while True:
            it += 1
            t_it = time.time()
            finished, st = pending
            t_roll = pending_roll
            _ddp_league_sync(st)
            _ddp_checks(it, st)
            curric_record(finished, st, it)
            curric_push(it, st)
            curric_check_push_seen(it, st)
            # Start the next batch in the background (actor weights: one update behind
            # after the first sync); the last --iters iteration has no next batch.
            task = None
            if not (args.iters and it >= args.iters):
                task = CollectTask(col, args.rollout).start()
            vrpo_info: dict = {}
            batch_ok = True
            try:
                if args.adv == "vrpo":
                    slots, adv, ret, adv_n, vrpo_info = vrpo_targets_batch(store, finished, args.lam)
                else:
                    slots, adv, ret, adv_n = gae_batch(store, finished, args.lam)
            except RuntimeError:
                batch_ok = False
                slots = np.zeros(0, np.int64)
                adv = ret = adv_n = np.zeros(0, np.float32)
            dfail_if_any(not batch_ok, lambda: f"iteration {it}: non-finite Q/Vbar/advantage/return")
            if is_dist:
                _gmean, _gstd, _ = distmod.global_adv_stats(adv, dev)
                adv_n = ((adv - _gmean) / (_gstd + 1e-8)).astype(np.float32)
            else:
                _gstd = float(adv.std())
            if cur is not None and args.curric_drop_solved:
                slots, adv, ret, adv_n, _n_drop = curric_filter(
                    finished, st.get("fin_starts", []), slots, adv, ret, adv_n, it)
                if dist_rank == 0 and _n_drop:
                    print(f"[curriculum it {it}] dropped {_n_drop} solved-start samples", flush=True)
            dfail_if_any(len(slots) == 0, lambda: f"iteration {it}: collected 0 samples")
            warm = it <= args.value_warmup
            upd_ok = True
            try:
                agg_mean, t_update = ppo_update(
                    model, upd_model, opt, vopt, store, slots, adv_n, ret, rng, dev,
                    clip=args.clip, vf_coef=args.vf_coef, kl_coef=args.kl_coef,
                    ent_coef=args.ent_coef, epochs=args.epochs, minibatch=args.minibatch,
                    warm=warm, precision=args.precision, max_upload_gb=args.max_upload_gb,
                    compile=args.compile, critic=critic, copt=copt,
                    priv_critic=bool(args.priv_critic), adv=args.adv, upd_critic=upd_critic,
                    hint=hint)
            except RuntimeError:
                upd_ok = False
                agg_mean, t_update = {}, 0.0
            dfail_if_any(not upd_ok, lambda: f"iteration {it}: non-finite loss")
            if is_dist and it == 1:
                check_params_match("iteration 1")
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

            _kl_zero = (not warm and args.kl_coef > 0 and agg_mean.get("kl_bc") is not None
                        and agg_mean["kl_bc"] == 0.0)
            dfail_if_any(_kl_zero,
                         lambda: f"iteration {it}: kl_bc exactly 0.0 with --kl-coef {args.kl_coef}")
            dt = time.time() - t_it
            # Global numbers for rank 0's log line (one SUM collective + sps min/max).
            _sps_local = len(slots) / max(dt, 1e-9)
            if is_dist:
                _lkeys = sorted(agg_mean)
                _lsum = _ddp_sum([float(agg_mean[k]) for k in _lkeys])
                agg_mean = {k: v / dist_world for k, v in zip(_lkeys, _lsum)}
                _cs = _ddp_sum([float(len(slots)), float(st["games"]), float(st["trunc"]),
                                 float(st["decisions"]), float(st["bc_games"]), float(st["bc_wins"]),
                                 float(st["dags_games"]),
                                 float(st["dags_wr"]) * float(st["dags_games"]),
                                 float(np.sum(st["lengths"])) if st["lengths"] else 0.0,
                                 float(len(st["lengths"])), float(ret.sum()) if len(ret) else 0.0,
                                 float(len(ret))])
                (_g_samples, _g_games, _g_trunc, _g_dec, _g_bcg, _g_bcw, _g_dg, _g_dw,
                 _g_sumlen, _g_nlen, _g_sumret, _g_nret) = _cs
                _g_adv_std, _g_ret_mean = _gstd, _g_sumret / max(_g_nret, 1)
                _sps_min = distmod.min_scalar(_sps_local, dev)
                _sps_max = distmod.max_scalar(_sps_local, dev)
            else:
                _g_samples, _g_games, _g_trunc, _g_dec = (float(len(slots)), float(st["games"]),
                                                          float(st["trunc"]), float(st["decisions"]))
                _g_bcg, _g_bcw = float(st["bc_games"]), float(st["bc_wins"])
                _g_dg = float(st["dags_games"])
                _g_dw = float(st["dags_wr"]) * float(st["dags_games"])
                _g_adv_std = _gstd
                _g_ret_mean = float(ret.mean()) if len(ret) else 0.0
                _g_sumlen = float(np.sum(st["lengths"])) if st["lengths"] else 0.0
                _g_nlen = float(len(st["lengths"]))
                _sps_min = _sps_max = _sps_local
            if dist_rank == 0:
                tot["games"] += _g_games
                tot["trunc"] += _g_trunc
                tot["decisions"] += _g_dec
                rec = {
                    "iter": it, "warmup": warm, "samples": _g_samples, "games": _g_games,
                    "trunc_rate": round(_g_trunc / max(_g_games, 1), 3),
                    "mean_game_decisions": round(_g_sumlen / max(_g_nlen, 1), 1),
                    "train_vs_bc_sampled": round(_g_bcw / max(_g_bcg, 1), 3), "bc_games": _g_bcg,
                    "dags_games": _g_dg, "dags_wr": round(_g_dw / max(_g_dg, 1), 3),
                    "dags_bad": st["dags_bad"],
                    **({"curric": cur.stats()} if cur is not None else {}),
                    **({"hint_rows": hint["rows"]} if hint is not None else {}),
                    "adv_std": round(float(_g_adv_std), 3), "ret_mean": round(float(_g_ret_mean), 3),
                    "dps": round(_g_dec / t_roll), "rollout_s": round(t_roll, 1), "iter_s": round(dt, 1),
                    "t_env": round(st["t_env"], 1), "t_fwd": round(st["t_fwd"], 1),
                    "update_s": round(t_update, 1), "sps": round(_g_samples / dt, 1),
                    "store_free": store.free, "hours": round((time.time() - t_train) / 3600, 3),
                    "overlap": True, "wait_s": round(wait_s, 2),
                    **({"world_size": dist_world, "sps_min": round(_sps_min, 1),
                        "sps_max": round(_sps_max, 1)} if is_dist else {}),
                    **({"vbar_mean": round(vrpo_info["vbar_mean"], 4),
                        "adv_diff": round(vrpo_info["adv_diff"], 4)} if vrpo_info else {}),
                    **({"league": st["league"], "t_ga": round(st["t_ga"], 1)} if league is not None else {}),
                    **{k: round(v, 4) for k, v in agg_mean.items()},
                }  # fmt: skip
                print(json.dumps(rec), flush=True)
                if league is not None:
                    print(league_line(it, st), flush=True)
                assert log is not None
                log.write(json.dumps(rec) + "\n")
                log.flush()
            _local_done = (time.time() - t_train) / 3600 >= args.hours or (args.iters and it >= args.iters)
            out_of_time = distmod.bcast_flag(_local_done if dist_rank == 0 else False)
            if args.eval_mode == "async" and dist_rank == 0:
                assert asev is not None
                relay(asev.poll())
            if it % args.eval_every == 0 or out_of_time:
                if is_dist:
                    check_params_match(f"eval it {it}")
                if dist_rank == 0:
                    save_last(it)
                    save_curriculum()
                    if args.eval_mode == "sync":
                        evaluate(it)
                        recorded.add(it)
                        eval_started.add(it)
                    else:
                        assert asev is not None
                        ck = args.out / f"iter-{it:04d}.pt"
                        torch.save(policy_state(model), ck)
                        save_critic_pt()
                        if dev.type == "cuda":  # the eval subprocess loads 2 nets on the same GPU: give back the update's
                            torch.cuda.empty_cache()  # cached blocks first (A10 it 80 eval OOM, NOTEBOOK 2026-10-07 12:20)
                        if asev.maybe_start(it, ck):
                            eval_started.add(it)
                        else:
                            print(f"[eval it {it}] skipped (previous eval running)", flush=True)
                if is_dist and args.eval_mode == "sync":
                    distmod.barrier()  # other ranks wait while rank 0 evaluates
            if out_of_time:
                if task is not None:
                    # Speculatively collected one batch too many: discard it, free its slots.
                    for s, _ in pending[0]:
                        store.release(s)
                if args.eval_mode == "async" and dist_rank == 0:
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
        dfail_if_any(not finished, lambda: f"iteration {it}: collected 0 samples")
        _ddp_league_sync(st)
        _ddp_checks(it, st)
        curric_record(finished, st, it)
        curric_push(it, st)
        curric_check_push_seen(it, st)

        # GAE per finished seat trajectory, then PPO epochs
        # (VRPO: expected-SARSA(λ) targets from the stored rollout-time π/Q instead)
        vrpo_info2: dict = {}
        batch_ok2 = True
        try:
            if args.adv == "vrpo":
                slots, adv, ret, adv_n, vrpo_info2 = vrpo_targets_batch(store, finished, args.lam)
            else:
                slots, adv, ret, adv_n = gae_batch(store, finished, args.lam)
        except RuntimeError:
            batch_ok2 = False
            slots = np.zeros(0, np.int64)
            adv = ret = adv_n = np.zeros(0, np.float32)
        dfail_if_any(not batch_ok2, lambda: f"iteration {it}: non-finite Q/Vbar/advantage/return")
        if is_dist:
            _gmean2, _gstd2, _ = distmod.global_adv_stats(adv, dev)
            adv_n = ((adv - _gmean2) / (_gstd2 + 1e-8)).astype(np.float32)
        else:
            _gstd2 = float(adv.std())
        if cur is not None and args.curric_drop_solved:
            slots, adv, ret, adv_n, _n_drop = curric_filter(
                finished, st.get("fin_starts", []), slots, adv, ret, adv_n, it)
            if dist_rank == 0 and _n_drop:
                print(f"[curriculum it {it}] dropped {_n_drop} solved-start samples", flush=True)
        dfail_if_any(len(slots) == 0, lambda: f"iteration {it}: collected 0 samples")
        warm = it <= args.value_warmup
        upd_ok2 = True
        try:
            agg_mean, t_update = ppo_update(
                model, upd_model, opt, vopt, store, slots, adv_n, ret, rng, dev,
                clip=args.clip, vf_coef=args.vf_coef, kl_coef=args.kl_coef,
                ent_coef=args.ent_coef, epochs=args.epochs, minibatch=args.minibatch,
                warm=warm, precision=args.precision, max_upload_gb=args.max_upload_gb,
                compile=args.compile, critic=critic, copt=copt,
                priv_critic=bool(args.priv_critic), adv=args.adv, upd_critic=upd_critic,
                hint=hint)
        except RuntimeError:
            upd_ok2 = False
            agg_mean, t_update = {}, 0.0
        dfail_if_any(not upd_ok2, lambda: f"iteration {it}: non-finite loss")
        if is_dist and it == 1:
            check_params_match("iteration 1")
        note_qev(it, agg_mean, warm)
        for s, _ in finished:
            store.release(s)
        if league is not None:
            league_snapshot(model, league, it, args.snap_every, args.league_size)

        _kl_zero2 = (not warm and args.kl_coef > 0 and agg_mean.get("kl_bc") is not None
                     and agg_mean["kl_bc"] == 0.0)
        dfail_if_any(_kl_zero2,
                     lambda: f"iteration {it}: kl_bc exactly 0.0 with --kl-coef {args.kl_coef}")
        dt = time.time() - t_it
        _sps_local2 = len(slots) / max(dt, 1e-9)
        if is_dist:
            _lkeys2 = sorted(agg_mean)
            _lsum2 = _ddp_sum([float(agg_mean[k]) for k in _lkeys2])
            agg_mean = {k: v / dist_world for k, v in zip(_lkeys2, _lsum2)}
            _cs2 = _ddp_sum([float(len(slots)), float(st["games"]), float(st["trunc"]),
                              float(st["decisions"]), float(st["bc_games"]), float(st["bc_wins"]),
                              float(st["dags_games"]),
                              float(st["dags_wr"]) * float(st["dags_games"]),
                              float(np.sum(st["lengths"])) if st["lengths"] else 0.0,
                              float(len(st["lengths"])), float(ret.sum()) if len(ret) else 0.0,
                              float(len(ret))])
            (_g_samples, _g_games, _g_trunc, _g_dec, _g_bcg, _g_bcw, _g_dg, _g_dw,
             _g_sumlen, _g_nlen, _g_sumret, _g_nret) = _cs2
            _g_adv_std, _g_ret_mean = _gstd2, _g_sumret / max(_g_nret, 1)
            _sps_min = distmod.min_scalar(_sps_local2, dev)
            _sps_max = distmod.max_scalar(_sps_local2, dev)
        else:
            _g_samples, _g_games, _g_trunc, _g_dec = (float(len(slots)), float(st["games"]),
                                                      float(st["trunc"]), float(st["decisions"]))
            _g_bcg, _g_bcw = float(st["bc_games"]), float(st["bc_wins"])
            _g_dg = float(st["dags_games"])
            _g_dw = float(st["dags_wr"]) * float(st["dags_games"])
            _g_adv_std = _gstd2
            _g_ret_mean = float(ret.mean()) if len(ret) else 0.0
            _g_sumlen = float(np.sum(st["lengths"])) if st["lengths"] else 0.0
            _g_nlen = float(len(st["lengths"]))
            _sps_min = _sps_max = _sps_local2
        if dist_rank == 0:
            tot["games"] += _g_games
            tot["trunc"] += _g_trunc
            tot["decisions"] += _g_dec
            rec = {
                "iter": it, "warmup": warm, "samples": _g_samples, "games": _g_games,
                "trunc_rate": round(_g_trunc / max(_g_games, 1), 3),
                "mean_game_decisions": round(_g_sumlen / max(_g_nlen, 1), 1),
                "train_vs_bc_sampled": round(_g_bcw / max(_g_bcg, 1), 3), "bc_games": _g_bcg,
                "dags_games": _g_dg, "dags_wr": round(_g_dw / max(_g_dg, 1), 3),
                "dags_bad": st["dags_bad"],
                **({"curric": cur.stats()} if cur is not None else {}),
                **({"hint_rows": hint["rows"]} if hint is not None else {}),
                "adv_std": round(float(_g_adv_std), 3), "ret_mean": round(float(_g_ret_mean), 3),
                "dps": round(_g_dec / t_roll), "rollout_s": round(t_roll, 1), "iter_s": round(dt, 1),
                "t_env": round(st["t_env"], 1), "t_fwd": round(st["t_fwd"], 1),
                    "update_s": round(t_update, 1), "sps": round(_g_samples / dt, 1),
                    "store_free": store.free, "hours": round((time.time() - t_train) / 3600, 3),
                    **({"world_size": dist_world, "sps_min": round(_sps_min, 1),
                        "sps_max": round(_sps_max, 1)} if is_dist else {}),
                    **({"vbar_mean": round(vrpo_info2["vbar_mean"], 4),
                        "adv_diff": round(vrpo_info2["adv_diff"], 4)} if vrpo_info2 else {}),
                    **({"league": st["league"], "t_ga": round(st["t_ga"], 1)} if league is not None else {}),
                    **{k: round(v, 4) for k, v in agg_mean.items()},
                }  # fmt: skip
            print(json.dumps(rec), flush=True)
            if league is not None:
                print(league_line(it, st), flush=True)
            assert log is not None
            log.write(json.dumps(rec) + "\n")
            log.flush()
        _local_done2 = (time.time() - t_train) / 3600 >= args.hours or (args.iters and it >= args.iters)
        out_of_time = distmod.bcast_flag(_local_done2 if dist_rank == 0 else False)
        if args.eval_mode == "async" and dist_rank == 0:
            assert asev is not None
            relay(asev.poll())
        if it % args.eval_every == 0 or out_of_time:
            if is_dist:
                check_params_match(f"eval it {it}")
            if dist_rank == 0:
                save_last(it)
                save_curriculum()
                if args.eval_mode == "sync":
                    evaluate(it)
                    recorded.add(it)
                    eval_started.add(it)
                else:
                    assert asev is not None
                    ck = args.out / f"iter-{it:04d}.pt"
                    torch.save(policy_state(model), ck)
                    save_critic_pt()
                    if dev.type == "cuda":  # the eval subprocess loads 2 nets on the same GPU: give back the update's
                        torch.cuda.empty_cache()  # cached blocks first (A10 it 80 eval OOM, NOTEBOOK 2026-10-07 12:20)
                    if asev.maybe_start(it, ck):
                        eval_started.add(it)
                    else:
                        print(f"[eval it {it}] skipped (previous eval running)", flush=True)
            if is_dist and args.eval_mode == "sync":
                distmod.barrier()  # other ranks wait while rank 0 evaluates
        if out_of_time:
            if args.eval_mode == "async" and dist_rank == 0:
                assert asev is not None
                relay(asev.poll())
                relay(asev.wait())
                if it not in recorded and it not in eval_started:
                    evaluate(it)  # final eval was skipped: run it blocking so it is never missing
                    recorded.add(it)
                    eval_started.add(it)
            break
    if dist_rank == 0:
        print(f"done: {it} iterations, {tot['games']} games, {tot['decisions']} decisions, "
              f"{(time.time() - t_start) / 3600:.2f} h", flush=True)  # fmt: skip
    distmod.teardown()


if __name__ == "__main__":
    main()
