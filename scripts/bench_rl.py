"""GPU throughput benchmark for every stage of RL (scripts/train_rl.py), run on a vast box.

Measures, stage by stage: (a) env steps/s, (b) inference rows/s, (c) end-to-end
rollout decisions/s, (d) PPO update samples/s, under fp32/bf16 x trim on/off grids.

    uv run python scripts/bench_rl.py --ckpt runs/<run>/bc/best.pt --out runs/<name>/bench.jsonl

Every measurement prints one JSON line and appends it to --out:
{"section", "config", "rate", "unit", "seconds", "peak_gpu_mb", "rss_mb", ...}.
Exits 2 with a line starting `ALERT bench_rl:` if any rate is 0, a loss is NaN,
or the capture pool has fewer than 1024 rows.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_rl import load_net  # architecture, as in train_rl

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import VecEnv
from ygo_zero.engine.duel import Engine
from ygo_zero.eval.bc_policy import _run_args
from ygo_zero.model.policy import PolicyNet
from ygo_zero.rl.ppo import SlotStore, gather, masked_log_softmax, ppo_loss, to_torch
from ygo_zero.rl.rollout import Collector

ROOT = Path(__file__).resolve().parents[1]
POOL_N = 8192  # capture pool rows (numpy, host), shared by infer and update
SECTIONS = ("env", "infer", "rollout", "update", "profile")


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)


def _peak_mb(dev: torch.device) -> float:
    if dev.type == "cuda":
        return round(torch.cuda.max_memory_allocated(dev) / 1e6, 1)
    return 0.0


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _precisions(dev: torch.device) -> list[str]:
    return ["fp32"] if dev.type != "cuda" else ["fp32", "bf16"]


def _autocast(dev: torch.device, prec: str):
    """No-op context for fp32 / cpu, bf16 autocast on cuda."""
    if prec == "bf16" and dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _random_legal(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform random legal option index per row (mask [N, 64] bool)."""
    counts = mask.sum(1).clip(min=1)
    r = (rng.random(len(mask)) * counts).astype(np.int64)
    return np.argmax(mask.cumsum(1) > r[:, None], axis=1).astype(np.int32)


def _measure(fn, seconds: float, dev: torch.device):
    """3 warm-up iterations, then timed loop; fn() returns units done. Returns (units, dt)."""
    for _ in range(3):
        fn()
    _sync(dev)
    t0 = time.perf_counter()
    tot = 0
    while True:
        tot += fn()
        if time.perf_counter() - t0 >= seconds:
            break
    _sync(dev)
    return tot, time.perf_counter() - t0


def _make_env_step(vec: VecEnv, buf: dict, rng: np.random.Generator, nenv: int):
    """One VecEnv step with bench_vecenv-style random actions; returns agent steps done."""

    def step() -> int:
        counts = buf["action_mask"].sum(1)
        vec.step((rng.random(nenv) * counts).astype(np.int32))
        return nenv

    return step


def _make_infer_step(model: PolicyNet, views: list, batch: int, prec: str, dev: torch.device):
    """One device-resident forward_value; returns rows done."""
    it = [0]
    nb = len(views)

    def fwd() -> int:
        b = views[it[0] % nb]
        it[0] += 1
        with _autocast(dev, prec):
            model.forward_value(b)
        return batch

    return fwd


def _make_h2d_step(model: PolicyNet, pool: dict, batch: int, prec: str, dev: torch.device):
    """One to_torch + forward_value from host rows; returns rows done."""
    it = [0]
    n = len(pool["action_mask"])

    def h2d() -> int:
        j = (it[0] * batch) % n
        it[0] += 1
        sl = {k: v[j:j + batch] if j + batch <= n else np.concatenate([v[j:], v[:j + batch - n]])
              for k, v in pool.items()}
        with _autocast(dev, prec):
            model.forward_value(to_torch(sl, dev))
        return batch

    return h2d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None, help="BC checkpoint (architecture from its metrics.jsonl)")
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seconds", type=float, default=10.0, help="per measurement")
    ap.add_argument("--out", type=Path, required=True, help="jsonl path (appended)")
    ap.add_argument("--sections", nargs="*", default=list(SECTIONS))
    ap.add_argument("--smoke", action="store_true", help="tiny PolicyNet from synthetic emb, CPU, ~1 s per measurement")
    args = ap.parse_args()
    bad = [s for s in args.sections if s not in SECTIONS]
    if bad:
        print(f"ALERT bench_rl: unknown sections {bad}", flush=True)
        raise SystemExit(2)
    if args.smoke:
        args.seconds = 1.0
    seconds = float(args.seconds)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pre_lines = len(out.read_text().splitlines()) if out.is_file() else 0  # append mode: only check this run's lines
    fh = out.open("a")

    def emit(rec: dict) -> None:
        print(json.dumps(rec), flush=True)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    decks = [load_ydk(p) for p in sorted((ROOT / "decks/pool").glob("*.ydk"))]
    assert decks, "no decks in decks/pool"

    if args.smoke:
        codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
        rng_e = np.random.default_rng(0)
        n = len(codes)
        emb = {"codes": np.array(codes), "card": rng_e.normal(size=(n, 8)).astype(np.float32),
               "effects": rng_e.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
        model = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, value=True)
        hist, known = 0, False
        env_envs, infer_batches, rollout_envs, update_mbs = [16], [64], [16], [64]
        h2d_batch = 64
        store_cap = 65536
    else:
        if args.ckpt is None:
            print("ALERT bench_rl: --ckpt is required without --smoke", flush=True)
            raise SystemExit(2)
        a = _run_args(args.ckpt)
        hist, known = int(a.get("hist", 0) or 0), bool(a.get("known", False))
        model = load_net(args.ckpt, dict(np.load(args.emb)), a, value=True)
        env_envs, infer_batches, rollout_envs, update_mbs = [256, 1024], [256, 1024, 4096], [256, 1024], [256, 1024, 4096]  # fmt: skip
        h2d_batch = 1024
        store_cap = 400_000  # train_rl's --store default: open trajectories only, released on game end
    model.to(dev)
    engine = Engine()

    # Capture: random legal play (256 envs) until POOL_N gathered rows (numpy, host).
    pool: dict[str, np.ndarray] | None = None
    if "infer" in args.sections or "update" in args.sections or "profile" in args.sections:
        rng = np.random.default_rng(999)
        vec = VecEnv(engine, decks, 256, args.threads, seed=999)
        buf = vec.reset()
        parts = []
        total = 0
        while total < POOL_N:
            parts.append({k: np.array(v) for k, v in gather(buf, np.arange(256), hist, known).items()})
            total += 256
            vec.step(_random_legal(buf["action_mask"], rng))
            vec.pop_finished()
        pool = {k: np.concatenate([p[k] for p in parts])[:POOL_N] for k in parts[0]}
        if len(pool["action_mask"]) < 1024:
            print(f"ALERT bench_rl: pool has {len(pool['action_mask'])} rows (< 1024)", flush=True)
            raise SystemExit(2)

    nan_loss = False

    if "env" in args.sections:
        for nenv in env_envs:
            rng = np.random.default_rng(7)
            vec = VecEnv(engine, decks, nenv, args.threads, seed=123)
            buf = vec.reset()
            _reset_peak(dev)
            units, dt = _measure(_make_env_step(vec, buf, rng, nenv), seconds, dev)
            emit({"section": "env", "config": {"envs": nenv, "threads": args.threads},
                  "rate": units / dt, "unit": "agent_steps/s", "seconds": round(dt, 3),
                  "peak_gpu_mb": _peak_mb(dev), "rss_mb": _rss_mb()})  # fmt: skip

    if "infer" in args.sections:
        assert pool is not None
        model.eval()
        pool_dev = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in pool.items()}
        with torch.no_grad():
            for batch in infer_batches:
                nb = max(1, POOL_N // batch)
                views = [{k: v[i * batch:(i + 1) * batch] for k, v in pool_dev.items()} for i in range(nb)]
                for trim in (False, True):
                    model.trim = trim
                    for prec in _precisions(dev):
                        _reset_peak(dev)
                        units, dt = _measure(_make_infer_step(model, views, batch, prec, dev), seconds, dev)
                        emit({"section": "infer", "config": {"batch": batch, "trim": trim, "precision": prec},
                              "rate": units / dt, "unit": "rows/s", "seconds": round(dt, 3),
                              "peak_gpu_mb": _peak_mb(dev), "rss_mb": _rss_mb()})  # fmt: skip
            # one extra config with host-to-device included each iteration
            trim = True
            prec = "bf16" if dev.type == "cuda" else "fp32"
            model.trim = trim
            _reset_peak(dev)
            units, dt = _measure(_make_h2d_step(model, pool, h2d_batch, prec, dev), seconds, dev)
            emit({"section": "infer", "config": {"batch": h2d_batch, "trim": trim, "precision": prec, "h2d": True},
                  "rate": units / dt, "unit": "rows/s", "seconds": round(dt, 3),
                  "peak_gpu_mb": _peak_mb(dev), "rss_mb": _rss_mb()})  # fmt: skip

    if "rollout" in args.sections:
        model.eval()
        ref_roll = copy.deepcopy(model).eval()
        for p in ref_roll.parameters():
            p.requires_grad_(False)
        tgt = 256 if args.smoke else 2048
        for nenv in rollout_envs:
            for trim, use_bf16 in ((False, False), (True, True)):
                prec = "bf16" if (use_bf16 and dev.type == "cuda") else "fp32"
                model.trim = trim
                ref_roll.trim = trim
                vec = VecEnv(engine, decks, nenv, args.threads, seed=4242)
                vec.set_max_decisions(1500)  # train_rl's --max-decisions default
                store = SlotStore(store_cap, gather(vec.buf, np.arange(1), hist, known))
                col = Collector(vec, model, ref_roll, store, hist, known, dev, 0.0, -0.5,
                                max(1, nenv // 4), np.random.default_rng(4242), prec)
                warm, _ = col.collect(min(tgt, 64))
                for s, _ in warm:
                    store.release(s)
                _reset_peak(dev)
                _sync(dev)
                t0 = time.perf_counter()
                units = t_env = t_fwd = 0.0
                units_n = 0
                while True:
                    finished, st = col.collect(tgt)
                    units_n += st["decisions"]
                    t_env += st["t_env"]
                    t_fwd += st["t_fwd"]
                    for s, _ in finished:
                        store.release(s)
                    if time.perf_counter() - t0 >= seconds:
                        break
                _sync(dev)
                dt = time.perf_counter() - t0
                units = units_n
                emit({"section": "rollout", "config": {"envs": nenv, "trim": trim, "precision": prec},
                      "rate": units / dt, "unit": "decisions/s", "seconds": round(dt, 3),
                      "peak_gpu_mb": _peak_mb(dev), "rss_mb": _rss_mb(),
                      "t_env": round(t_env, 3), "t_fwd": round(t_fwd, 3)})  # fmt: skip

    if "update" in args.sections:
        assert pool is not None
        model.train()
        # frozen copy of the learner: its log-probs are cached once per trim (no ref forward in the update)
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        counts = pool["action_mask"].sum(1).clip(min=1)
        log1p = np.log(1.0 / counts).astype(np.float32)
        rng = np.random.default_rng(2026)
        for mb in update_mbs:
            for trim in (False, True):
                model.trim = trim
                ref.trim = trim
                with torch.no_grad():
                    parts = []
                    for j in range(0, len(pool["action_mask"]), 1024):
                        sl = {k: v[j:j + 1024] for k, v in pool.items()}
                        parts.append(masked_log_softmax(ref(to_torch(sl, dev))).float().cpu().numpy())
                    ref_pool = np.concatenate(parts).astype(np.float16)  # cached KL reference (POOL_N, 64)
                for prec in _precisions(dev):
                    for kl in ("on", "off"):
                        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
                        try:
                            _update_one(mb, trim, prec, kl, model, ref_pool if kl == "on" else None,
                                        pool, log1p, rng, opt, dev, seconds, emit, nan_flag := [False])
                            if nan_flag[0]:
                                nan_loss = True
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            emit({"section": "update",
                                  "config": {"minibatch": mb, "trim": trim, "precision": prec, "kl": kl,
                                             "fused": False, "compile": False},
                                  "rate": None, "unit": "samples/s", "seconds": 0.0,
                                  "peak_gpu_mb": _peak_mb(dev), "rss_mb": _rss_mb(), "skipped": "oom"})  # fmt: skip
        # Speed knobs, one point only (mb1024 + trim + bf16 + KL on): fused x compile.
        # (Smoke: the smoke minibatch + trim + fp32 + KL on, same idea, no grid blowup.)
        mb_f = 1024 if 1024 in update_mbs else update_mbs[-1]
        prec_f = "bf16" if dev.type == "cuda" else "fp32"
        model.trim = True
        ref.trim = True
        with torch.no_grad():
            parts = []
            for j in range(0, len(pool["action_mask"]), 1024):
                sl = {k: v[j:j + 1024] for k, v in pool.items()}
                parts.append(masked_log_softmax(ref(to_torch(sl, dev))).float().cpu().numpy())
            ref_pool_f = np.concatenate(parts).astype(np.float16)
        for fused in (False, True):
            for compiled in (False, True):
                try:
                    kw = {"fused": True} if fused and dev.type == "cuda" else {}
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, **kw)
                except (RuntimeError, TypeError, ValueError):
                    emit({"section": "update",
                          "config": {"minibatch": mb_f, "trim": True, "precision": prec_f, "kl": "on",
                                     "fused": fused, "compile": compiled},
                          "rate": None, "unit": "samples/s", "seconds": 0.0,
                          "peak_gpu_mb": _peak_mb(dev), "rss_mb": _rss_mb(), "skipped": "unsupported"})  # fmt: skip
                    continue
                try:
                    _update_one(mb_f, True, prec_f, "on", model, ref_pool_f, pool, log1p, rng, opt,
                                dev, seconds, emit, nan_flag := [False], fused=fused, compiled=compiled)
                    if nan_flag[0]:
                        nan_loss = True
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    emit({"section": "update",
                          "config": {"minibatch": mb_f, "trim": True, "precision": prec_f, "kl": "on",
                                     "fused": fused, "compile": compiled},
                          "rate": None, "unit": "samples/s", "seconds": 0.0,
                          "peak_gpu_mb": _peak_mb(dev), "rss_mb": _rss_mb(), "skipped": "oom"})  # fmt: skip

    if "profile" in args.sections:
        assert pool is not None
        _run_profile(model, pool, dev, args.smoke, out, emit)

    fh.close()
    problems = []
    for raw in out.read_text().splitlines()[pre_lines:]:
        rec = json.loads(raw)
        if "skipped" in rec or rec.get("section") == "profile":
            continue
        if not rec.get("rate") or rec["rate"] <= 0:
            problems.append(f"{rec.get('section')}/{rec.get('config')} rate={rec.get('rate')}")
    if nan_loss:
        problems.append("NaN loss")
    if problems:
        print(f"ALERT bench_rl: {'; '.join(problems)}", flush=True)
        raise SystemExit(2)


def _update_one(mb: int, trim: bool, prec: str, kl: str, model: PolicyNet, ref_pool, pool, log1p, rng,
                opt, dev: torch.device, seconds: float, emit, nan_flag: list,
                fused: bool = False, compiled: bool = False) -> None:
    n_pool = len(pool["action_mask"])
    model.trim = trim
    # Update forward only (train_rl --compile semantics); the eager model keeps the params/optimizer.
    upd = torch.compile(model, dynamic=True) if compiled else model
    if compiled:
        upd.train()
        try:
            upd.trim = trim
        except (AttributeError, TypeError):
            pass
    # On-device accumulation (no per-minibatch host sync): stats sums + a finite flag stay
    # on-device; converted once after the timed loop.
    sums: dict[str, torch.Tensor] = {}
    n_mb = [0]
    finite = [torch.ones((), dtype=torch.bool, device=dev)]

    def ustep() -> int:
        idx = rng.integers(0, n_pool, mb)
        obs_np = {k: v[idx] for k, v in pool.items()}
        act_np = _random_legal(obs_np["action_mask"], rng)
        olp_np = log1p[idx]
        adv_np = rng.standard_normal(mb).astype(np.float32)
        ret_np = rng.uniform(-1, 1, mb).astype(np.float32)
        tob = to_torch(obs_np, dev)
        act_t = torch.from_numpy(act_np).to(dev)
        olp_t = torch.from_numpy(olp_np).to(dev)
        adv_t = torch.from_numpy(adv_np).to(dev)
        ret_t = torch.from_numpy(ret_np).to(dev)
        # cached KL reference (None = no KL); no ref forward inside the timed update
        ref_t = torch.from_numpy(ref_pool[idx]).to(dev) if kl == "on" and ref_pool is not None else None
        opt.zero_grad(set_to_none=True)
        with _autocast(dev, prec):
            # train_rl defaults: clip 0.2, vf 0.5, kl 0.1, ent 0.003
            loss, st = ppo_loss(upd, ref_t, tob, act_t, olp_t, adv_t, ret_t, 0.2, 0.5, 0.1, 0.003)
        finite[0] = finite[0] & torch.isfinite(loss).detach()
        for k, v in st.items():
            t = v.detach()
            sums[k] = t.clone() if k not in sums else sums[k] + t
        n_mb[0] += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return mb

    _reset_peak(dev)
    units, dt = _measure(ustep, seconds, dev)
    if not bool(finite[0].item()):  # single host sync for the whole measurement
        nan_flag[0] = True
    if n_mb[0]:  # one sync per key for the whole measurement (values mirror train_rl's agg means)
        {k: float((v / n_mb[0]).item()) for k, v in sums.items()}
    emit({"section": "update",
          "config": {"minibatch": mb, "trim": trim, "precision": prec, "kl": kl,
                     "fused": fused, "compile": compiled},
          "rate": units / dt, "unit": "samples/s", "seconds": round(dt, 3),
          "peak_gpu_mb": _peak_mb(dev), "rss_mb": _rss_mb()})  # fmt: skip


def _run_profile(model: PolicyNet, pool: dict, dev: torch.device, smoke: bool, out: Path, emit) -> None:
    """torch.profiler over 20 update minibatches (mb1024/trim/bf16/cached-ref KL, 5 warm-up).

    Smoke (CPU, tiny net): mb64/fp32. Writes key_averages().table() to <out dir>/profile.txt
    plus one {"section": "profile", "top": [...]} jsonl line (top 15 ops).
    """
    mb = 64 if smoke else 1024
    prec = "bf16" if dev.type == "cuda" else "fp32"
    model.train()
    model.trim = True
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.trim = True
    with torch.no_grad():
        parts = []
        for j in range(0, len(pool["action_mask"]), 1024):
            sl = {k: v[j:j + 1024] for k, v in pool.items()}
            parts.append(masked_log_softmax(ref(to_torch(sl, dev))).float().cpu().numpy())
        ref_pool = np.concatenate(parts).astype(np.float16)
    counts = pool["action_mask"].sum(1).clip(min=1)
    log1p = np.log(1.0 / counts).astype(np.float32)
    rng = np.random.default_rng(202607)
    n_pool = len(pool["action_mask"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def mb_step() -> None:
        idx = rng.integers(0, n_pool, mb)
        obs_np = {k: v[idx] for k, v in pool.items()}
        act_np = _random_legal(obs_np["action_mask"], rng)
        tob = to_torch(obs_np, dev)
        act_t = torch.from_numpy(act_np).to(dev)
        olp_t = torch.from_numpy(log1p[idx]).to(dev)
        adv_t = torch.from_numpy(rng.standard_normal(mb).astype(np.float32)).to(dev)
        ret_t = torch.from_numpy(rng.uniform(-1, 1, mb).astype(np.float32)).to(dev)
        ref_t = torch.from_numpy(ref_pool[idx]).to(dev)
        opt.zero_grad(set_to_none=True)
        with _autocast(dev, prec):
            loss, _ = ppo_loss(model, ref_t, tob, act_t, olp_t, adv_t, ret_t, 0.2, 0.5, 0.1, 0.003)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(5):  # warm-up (compile/recompile + autotune settle before profiling)
        mb_step()
    acts = [torch.profiler.ProfilerActivity.CPU]
    if dev.type == "cuda":
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=acts) as prof:
        for _ in range(20):
            mb_step()
    _sync(dev)
    ka = prof.key_averages()
    sort_by = "cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
    try:
        table = ka.table(sort_by=sort_by, row_limit=30)
    except (AssertionError, KeyError, RuntimeError):
        table = ka.table(sort_by="cpu_time_total", row_limit=30)

    def _cuda_ms(e) -> float:
        for attr in ("cuda_time_total", "device_time_total"):
            v = getattr(e, attr, None)
            if v:
                return round(float(v) / 1000, 3)
        return 0.0

    on_cuda = dev.type == "cuda"
    rows = sorted(ka, key=lambda e: (_cuda_ms(e) if on_cuda else float(e.cpu_time_total)), reverse=True)
    top = [[e.key, _cuda_ms(e), round(float(e.cpu_time_total) / 1000, 3), int(e.count)] for e in rows[:15]]
    prof_path = out.parent / "profile.txt"
    prof_path.write_text(f"# bench_rl profile: mb={mb} trim=True precision={prec} kl=on ({dev.type})\n" + table)
    emit({"section": "profile", "config": {"minibatch": mb, "trim": True, "precision": prec, "kl": "on"},
          "top": top, "profile_txt": str(prof_path)})  # fmt: skip


if __name__ == "__main__":
    main()
