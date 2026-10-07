"""Profile YgoAgentPolicy per-row cost: encoding (Python) vs TFLite invoke, and whether the 0546 model
accepts a batch dimension (resize_tensor_input). Run on a vast box (ML inference), never on the laptop:

    uv run --no-sync python scripts/audit/prof_ygoagent.py [--envs 64] [--steps 20]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import VecEnv
from ygo_zero.engine.duel import Engine
from ygo_zero.eval import ygoagent as Y

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--impl", type=Path, default=None, help="load YgoAgentPolicy from this file (A/B a change)")
    ap.add_argument("--compare-workers", type=int, default=0, help="check workers=N == workers=1 and time both")
    args = ap.parse_args()
    if args.compare_workers:
        compare(args)
        return
    decks = [load_ydk(p) for p in sorted((ROOT / "decks/pool").glob("*.ydk"))[:8]]
    vec = VecEnv(Engine(), decks, args.envs, 4, seed=1)
    buf = vec.reset()
    rng = np.random.default_rng(0)
    pol = Y.YgoAgentPolicy(seed=0, sample=True)
    rows = np.arange(args.envs)
    t_enc = t_call = 0.0
    n = 0
    for _ in range(args.steps):
        keys = rows.copy()
        t0 = time.perf_counter()
        for r in rows:
            cards_enc, index_map = Y.encode_cards(buf["cards"][r], pol.code_to_id)
            k = int(buf["action_mask"][r].sum())
            Y.encode_actions(buf["actions"][r], k, index_map, pol.code_to_id)
            Y.encode_global(buf["global"][r])
            Y.encode_history(buf["history"][r], buf["history_mask"][r], pol.code_to_id)
        t1 = time.perf_counter()
        pol(buf, rows, keys)
        t2 = time.perf_counter()
        t_enc += t1 - t0
        t_call += t2 - t1
        n += len(rows)
        acts = np.array([rng.integers(int(buf["action_mask"][r].sum())) for r in rows], np.int32)
        buf = vec.step(acts)
    print(f"rows {n}: full call {1e3 * t_call / n:.2f} ms/row, encode only {1e3 * t_enc / n:.2f} ms/row, "
          f"invoke+rest ≈ {1e3 * (t_call - t_enc) / n:.2f} ms/row")

    from concurrent.futures import ThreadPoolExecutor

    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    blob = Path(Y.MODEL_PATH).read_bytes()

    def zeros_in(it):
        for d in it.get_input_details():
            it.set_tensor(d["index"], np.zeros(d["shape"], d["dtype"]))

    # (a) batch without the XNNPack delegate
    for b in (1, 16):
        try:
            it = Interpreter(model_content=blob,
                             experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            for d in it.get_input_details():
                it.resize_tensor_input(d["index"], [b, *[int(x) for x in d["shape"][1:]]])
            it.allocate_tensors()
            zeros_in(it)
            t0 = time.perf_counter()
            for _ in range(10):
                it.invoke()
            print(f"no-xnnpack batch {b}: {1e3 * (time.perf_counter() - t0) / 10 / b:.2f} ms/row")
        except Exception as e:  # noqa: BLE001
            print(f"no-xnnpack batch {b} failed:", type(e).__name__, str(e)[:200])

    # (b) pool of K batch-1 XNNPack interpreters, 1 thread each, in a thread pool
    for k in (4, 8, 16):
        its = []
        for _ in range(k):
            it = Interpreter(model_content=blob, num_threads=1)
            it.allocate_tensors()
            zeros_in(it)
            its.append(it)
        per = 32

        def run(it):
            for _ in range(per):
                it.invoke()

        with ThreadPoolExecutor(k) as ex:
            t0 = time.perf_counter()
            list(ex.map(run, its))
            dt = time.perf_counter() - t0
        print(f"pool {k}x1thr: {1e3 * dt / (k * per):.2f} ms/row (wall per row)")

    # (c) default single interpreter for reference
    it = Interpreter(model_content=blob)
    it.allocate_tensors()
    zeros_in(it)
    t0 = time.perf_counter()
    for _ in range(64):
        it.invoke()
    print(f"single default: {1e3 * (time.perf_counter() - t0) / 64:.2f} ms/row")


def compare(args) -> None:
    import importlib.util

    mod = Y
    if args.impl:
        spec = importlib.util.spec_from_file_location("ygoagent_new", args.impl)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    decks = [load_ydk(p) for p in sorted((ROOT / "decks/pool").glob("*.ydk"))[:8]]
    res = {}
    for w in (1, args.compare_workers):
        vec = VecEnv(Engine(), decks, args.envs, 4, seed=1)
        buf = vec.reset()
        pol = mod.YgoAgentPolicy(seed=0, sample=True, workers=w)
        rows = np.arange(args.envs)
        acts_all, t = [], 0.0
        for _ in range(args.steps):
            t0 = time.perf_counter()
            a = pol(buf, rows, rows.copy())
            t += time.perf_counter() - t0
            acts_all.append(a.copy())
            buf = vec.step(a)
        res[w] = (np.concatenate(acts_all), t / (args.envs * args.steps))
        print(f"workers={w}: {1e3 * res[w][1]:.2f} ms/row")
    a1, aw = res[1][0], res[args.compare_workers][0]
    same = bool((a1 == aw).all())
    print(f"identical actions: {same} ({int((a1 != aw).sum())}/{len(a1)} differ); "
          f"speedup x{res[1][1] / res[args.compare_workers][1]:.2f}")
    if not same:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
