"""Throughput of the C++ VecEnv with random actions (agent steps/s) for several thread counts.

uv run python scripts/bench_vecenv.py --envs 64 --threads 1 2 4 8 --seconds 20
uv run python scripts/bench_vecenv.py --decks decks/pool --envs 256 --threads 4 --seconds 40 --mode sync async
"""

import argparse
import time
from pathlib import Path

import numpy as np

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import VecEnv
from ygo_zero.engine.duel import Engine


def _maybe_window(mode, t, t0, w0, wsteps, window, now):
    if now - w0 >= window:
        print(
            f"[{mode}] threads {t:2d} +{now - t0:5.1f}s: {wsteps / (now - w0):7.0f} agent steps/s",
            flush=True,
        )
        return now, 0
    return w0, wsteps


def _run_sync(vec, buf, envs, seconds, window, t, rng):
    steps = games = 0
    t0 = w0 = time.time()
    wsteps = 0
    while (now := time.time()) - t0 < seconds:
        counts = buf["action_mask"].sum(1)
        vec.step((rng.random(envs) * counts).astype(np.int32))
        steps += envs
        wsteps += envs
        games += len(vec.pop_finished())
        w0, wsteps = _maybe_window("sync", t, t0, w0, wsteps, window, time.time())
    dt = time.time() - t0
    print(f"[sync] threads {t:2d}: {steps / dt:7.0f} agent steps/s, {games / dt:5.1f} games/s", flush=True)
    return steps / dt


def _run_async(vec, buf, envs, batch, seconds, window, t, rng):
    steps = games = 0
    t0 = w0 = time.time()
    wsteps = 0
    while (now := time.time()) - t0 < seconds:
        ids = vec.recv(batch, envs)
        if len(ids) == 0:
            continue
        send_ids, send_acts = [], []
        for i in ids:
            i = int(i)
            if int(buf["player"][i]) == -1:
                continue
            count = int(buf["action_mask"][i].sum())
            send_ids.append(i)
            send_acts.append(int(rng.random() * count))
        if send_ids:
            vec.send(np.asarray(send_ids, dtype=np.int32), np.asarray(send_acts, dtype=np.int32))
            steps += len(send_ids)
            wsteps += len(send_ids)
        games += len(vec.pop_finished())
        w0, wsteps = _maybe_window("async", t, t0, w0, wsteps, window, time.time())
    dt = time.time() - t0
    print(f"[async] threads {t:2d}: {steps / dt:7.0f} agent steps/s, {games / dt:5.1f} games/s", flush=True)
    return steps / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--decks", type=Path, default=Path("../ygo-agent/assets/deck"))
    ap.add_argument("--mode", nargs="+", default=["sync", "async"], choices=["sync", "async"])
    ap.add_argument("--batch", type=int, default=None, help="recv min_ready for async (default envs // 4)")
    ap.add_argument("--window", type=float, default=10)
    args = ap.parse_args()
    batch = args.batch or max(1, args.envs // 4)
    engine = Engine()
    decks = [load_ydk(p) for p in sorted(args.decks.glob("*.ydk"))]
    rng = np.random.default_rng(0)
    for t in args.threads:
        for mode in args.mode:
            vec = VecEnv(engine, decks, args.envs, t, seed=1)
            if mode == "sync":
                _run_sync(vec, vec.reset(), args.envs, args.seconds, args.window, t, rng)
            else:
                _run_async(vec, vec.async_reset(), args.envs, batch, args.seconds, args.window, t, rng)


if __name__ == "__main__":
    main()
