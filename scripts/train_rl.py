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
from ygo_zero.rl.ppo import SlotStore, gae, gather, masked_log_softmax, ppo_loss, to_torch
from ygo_zero.rl.rollout import Collector

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


def load_net(ckpt: Path, emb: dict, a: dict, value: bool) -> PolicyNet:
    net = PolicyNet(emb, d=a.get("d", 256), layers=a.get("layers", 4), heads=a.get("heads", 4), dropout=0.0,
                    hist=int(a.get("hist", 0) or 0), known=bool(a.get("known", False)), value=value)  # fmt: skip
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("model", state)
    missing, unexpected = net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()},
                                              strict=False)  # fmt: skip
    assert not unexpected and all(k.startswith("value_head.") for k in missing), (missing, unexpected)
    return net


@torch.no_grad()
def sample(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    lp = masked_log_softmax(logits)
    p = lp.exp() * torch.isfinite(logits)
    a = torch.multinomial(p, 1).squeeze(1)
    return a.cpu().numpy().astype(np.int32), lp.gather(1, a[:, None]).squeeze(1).cpu().numpy()


def policy_state(model: PolicyNet) -> dict:
    return {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()
            if not k.startswith("value_head.")}  # fmt: skip


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


def ppo_update(model: PolicyNet, upd_model, opt, vopt, store: SlotStore, slots: np.ndarray,
               adv_n: np.ndarray, ret: np.ndarray, rng: np.random.Generator, dev: torch.device,
               *, clip: float, vf_coef: float, kl_coef: float, ent_coef: float, epochs: int,
               minibatch: int, warm: bool, precision: str, max_upload_gb: float, compile: bool) -> dict:
    """Upload-once + PPO epochs over one batch. Returns per-iteration mean stats (raises on non-finite)."""
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    per_slot = (sum(int(np.prod(v.shape[1:], dtype=np.int64)) * np.dtype(v.dtype).itemsize
                    for v in store.obs.values()) + 8 + 4 + 4 + 64 * 2 + 4 + 4)
    upload_once = per_slot * len(slots) <= max_upload_gb * 1e9
    obs_dev = act_dev = olp_dev = ref_dev = adv_dev = ret_dev = perm_dev = None
    if upload_once:
        obs_dev = {k: torch.from_numpy(np.ascontiguousarray(store.obs[k][slots])).to(dev)
                   for k in store.obs}
        act_dev = torch.from_numpy(np.ascontiguousarray(store.action[slots])).to(dev)
        olp_dev = torch.from_numpy(np.ascontiguousarray(store.logp[slots])).to(dev)
        ref_dev = torch.from_numpy(np.ascontiguousarray(store.ref_lp[slots])).to(dev)
        adv_dev = torch.from_numpy(np.ascontiguousarray(adv_n)).to(dev)
        ret_dev = torch.from_numpy(np.ascontiguousarray(ret)).to(dev)
        perm_dev = torch.from_numpy(rng.permutation(len(slots))).to(dev)
    t_up = time.time()
    model.train()
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
            else:
                mb = perm[k : k + minibatch]
                sl = slots[mb]
                obs = to_torch({key: v[sl] for key, v in store.obs.items()}, dev)
                act = torch.from_numpy(store.action[sl]).to(dev)
                olp = torch.from_numpy(store.logp[sl]).to(dev)
                rl = torch.from_numpy(store.ref_lp[sl]).to(dev) if kl_coef > 0 else None
                a_t = torch.from_numpy(adv_n[mb]).to(dev)
                r_t = torch.from_numpy(ret[mb]).to(dev)
            if warm:
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
            else:
                with _autocast(dev, precision):
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
    ap.add_argument("--p-bc", type=float, default=0.3, help="share of episodes vs the frozen BC policy")
    ap.add_argument("--rollout", type=int, default=32768, help="finished samples per PPO iteration")
    ap.add_argument("--store", type=int, default=400_000, help="sample slots (open + finished trajectories)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--value-warmup", type=int, default=2, help="first iterations train the value head only")
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
    model = load_net(args.init, emb, a, value=True).to(dev).eval()
    ref = load_net(args.init, emb, a, value=False).to(dev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    vparams = list(model.value_head.parameters())
    trunk = [p for n, p in model.named_parameters() if not n.startswith("value_head.")]
    _fused = {"fused": True} if args.fused_adam and dev.type == "cuda" else {}
    opt = torch.optim.AdamW([{"params": trunk}, {"params": vparams, "lr": 1e-4}], lr=args.lr,
                            weight_decay=0.0, **_fused)
    vopt = torch.optim.AdamW(model.value_head.parameters(), lr=3e-4, **_fused)
    # Update forward only: rollout Collector keeps the eager `model`; the compiled
    # module shares parameters with it, so optimizer steps and saved state_dicts are unchanged.
    upd_model = torch.compile(model, dynamic=True) if args.compile else model

    log = (args.out / "metrics.jsonl").open("a")
    arch = {k: a.get(k) for k in ("d", "layers", "heads", "hist", "known")}
    log.write(json.dumps({"args": {**arch, "rl": {k: str(v) for k, v in vars(args).items()}}, "eval_decks": ev_names,
                          "train_decks": len(train_decks)}) + "\n")  # fmt: skip
    log.flush()

    engine = Engine()

    def evaluate(it: int) -> dict:
        """Synchronous eval: save iter-NNNN.pt, run in-process, print + log the record."""
        ck = args.out / f"iter-{it:04d}.pt"
        torch.save(policy_state(model), ck)
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
    store = SlotStore(args.store, gather(vec.buf, np.arange(1), hist, known))
    actor = None
    col_rng = rng
    if args.overlap:
        # The collector runs in a background thread while the main thread updates `model`,
        # so it forwards a private actor copy (synced after each update, never mid-collect).
        # It also gets its own RNG: np Generators are not thread-safe and the main thread
        # keeps using `rng` for minibatch permutations during the update.
        actor = copy.deepcopy(model).eval()
        col_rng = np.random.default_rng(args.seed + 0x1234)
    col = Collector(vec, actor if actor is not None else model, ref, store, hist, known, dev,
                    args.p_bc, args.trunc_reward, args.recv_batch, col_rng, args.precision)
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
            # Start the next batch in the background (actor weights: one update behind
            # after the first sync); the last --iters iteration has no next batch.
            task = None
            if not (args.iters and it >= args.iters):
                task = CollectTask(col, args.rollout).start()
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
                    compile=args.compile)
            except RuntimeError:
                fail(f"iteration {it}: non-finite loss")
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
                **{k: round(v, 4) for k, v in agg_mean.items()},
            }  # fmt: skip
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            out_of_time = (time.time() - t_train) / 3600 >= args.hours or (args.iters and it >= args.iters)
            if args.eval_mode == "async":
                assert asev is not None
                relay(asev.poll())
            if it % args.eval_every == 0 or out_of_time:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it}, args.out / "last.pt")
                if args.eval_mode == "sync":
                    evaluate(it)
                    recorded.add(it)
                    eval_started.add(it)
                else:
                    assert asev is not None
                    ck = args.out / f"iter-{it:04d}.pt"
                    torch.save(policy_state(model), ck)
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

        # GAE per finished seat trajectory, then PPO epochs
        slots = np.concatenate([np.array(s, np.int64) for s, _ in finished])
        if len(slots) == 0:
            fail(f"iteration {it}: collected 0 samples")
        advs, rets = [], []
        for s, rew in finished:
            ad, rt = gae(store.value[np.array(s)], rew, args.lam)
            advs.append(ad)
            rets.append(rt)
        adv, ret = np.concatenate(advs), np.concatenate(rets)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        warm = it <= args.value_warmup
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        # upload the iteration's samples to the device ONCE, index minibatches there
        per_slot = (sum(int(np.prod(v.shape[1:], dtype=np.int64)) * np.dtype(v.dtype).itemsize
                        for v in store.obs.values()) + 8 + 4 + 4 + 64 * 2 + 4 + 4)
        upload_once = per_slot * len(slots) <= args.max_upload_gb * 1e9
        obs_dev = act_dev = olp_dev = ref_dev = adv_dev = ret_dev = perm_dev = None
        if upload_once:
            obs_dev = {k: torch.from_numpy(np.ascontiguousarray(store.obs[k][slots])).to(dev)
                       for k in store.obs}
            act_dev = torch.from_numpy(np.ascontiguousarray(store.action[slots])).to(dev)
            olp_dev = torch.from_numpy(np.ascontiguousarray(store.logp[slots])).to(dev)
            ref_dev = torch.from_numpy(np.ascontiguousarray(store.ref_lp[slots])).to(dev)
            adv_dev = torch.from_numpy(np.ascontiguousarray(adv_n)).to(dev)
            ret_dev = torch.from_numpy(np.ascontiguousarray(ret)).to(dev)
            perm_dev = torch.from_numpy(rng.permutation(len(slots))).to(dev)
        t_up = time.time()
        model.train()
        if args.compile:
            upd_model.train()
        # On-device accumulation: ppo_loss returns detached 0-d tensors (no per-minibatch
        # host sync); sums stay on-device and convert once per iteration below.
        agg_sum: dict[str, torch.Tensor] = {}
        agg_n = 0
        bad_loss = False
        for _ in range(args.epochs):
            finite_epoch = torch.ones((), dtype=torch.bool, device=dev)
            perm = rng.permutation(len(slots))
            mb_dev = torch.from_numpy(perm).to(dev) if upload_once else None
            for k in range(0, len(perm), args.minibatch):
                if upload_once:
                    idx = mb_dev[k : k + args.minibatch]
                    obs = {key: v[idx] for key, v in obs_dev.items()}
                    act, olp = act_dev[idx], olp_dev[idx]
                    rl = ref_dev[idx] if args.kl_coef > 0 else None
                    a_t = adv_dev[idx]
                    r_t = ret_dev[idx]
                else:
                    mb = perm[k : k + args.minibatch]
                    sl = slots[mb]
                    obs = to_torch({key: v[sl] for key, v in store.obs.items()}, dev)
                    act = torch.from_numpy(store.action[sl]).to(dev)
                    olp = torch.from_numpy(store.logp[sl]).to(dev)
                    rl = torch.from_numpy(store.ref_lp[sl]).to(dev) if args.kl_coef > 0 else None
                    a_t = torch.from_numpy(adv_n[mb]).to(dev)
                    r_t = torch.from_numpy(ret[mb]).to(dev)
                if warm:
                    with _autocast(dev, args.precision):
                        with torch.no_grad():  # value head only: no graph through the trunk
                            _, enc = upd_model._run(obs)
                        v = upd_model.value_head(enc[:, 0]).squeeze(-1)
                        loss = torch.nn.functional.mse_loss(v.float(), r_t)
                    finite_epoch = finite_epoch & torch.isfinite(loss).detach()
                    vopt.zero_grad(set_to_none=True)
                    loss.backward()
                    vopt.step()
                    s = {"vf": loss.detach()}
                else:
                    with _autocast(dev, args.precision):
                        loss, s = ppo_loss(upd_model, rl, obs, act, olp, a_t, r_t, args.clip, args.vf_coef,
                                           args.kl_coef, args.ent_coef)  # fmt: skip
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
            fail(f"iteration {it}: non-finite loss")
        # One host sync per key per iteration (same values as the old per-minibatch .item() means).
        agg_mean = {k: float((v / agg_n).item()) for k, v in agg_sum.items()} if agg_n else {}
        t_update = time.time() - t_up
        model.eval()
        for s, _ in finished:
            store.release(s)

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
            **{k: round(v, 4) for k, v in agg_mean.items()},
        }  # fmt: skip
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        out_of_time = (time.time() - t_train) / 3600 >= args.hours or (args.iters and it >= args.iters)
        if args.eval_mode == "async":
            assert asev is not None
            relay(asev.poll())
        if it % args.eval_every == 0 or out_of_time:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it}, args.out / "last.pt")
            if args.eval_mode == "sync":
                evaluate(it)
                recorded.add(it)
                eval_started.add(it)
            else:
                assert asev is not None
                ck = args.out / f"iter-{it:04d}.pt"
                torch.save(policy_state(model), ck)
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
