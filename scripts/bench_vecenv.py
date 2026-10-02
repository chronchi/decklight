"""Throughput of the C++ VecEnv with random actions (agent steps/s) for several thread counts.

uv run python scripts/bench_vecenv.py --envs 64 --threads 1 2 4 8 --seconds 20
"""

import argparse
import time
from pathlib import Path

import numpy as np

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import VecEnv
from ygo_zero.engine.duel import Engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--decks", type=Path, default=Path("../ygo-agent/assets/deck"))
    args = ap.parse_args()
    engine = Engine()
    decks = [load_ydk(p) for p in sorted(args.decks.glob("*.ydk"))]
    rng = np.random.default_rng(0)
    for t in args.threads:
        vec = VecEnv(engine, decks, args.envs, t, seed=1)
        buf = vec.reset()
        steps = games = 0
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            counts = buf["action_mask"].sum(1)
            vec.step((rng.random(args.envs) * counts).astype(np.int32))
            steps += args.envs
            games += len(vec.pop_finished())
        dt = time.time() - t0
        print(f"threads {t:2d}: {steps / dt:7.0f} agent steps/s, {games / dt:5.1f} games/s", flush=True)


if __name__ == "__main__":
    main()
