"""Lockstep-compare the C++ env against the Python reference over random games; then time the C++ env.

uv run python scripts/check_cpp_env.py --duels 100
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "engine"))

from test_cpp_env import lockstep

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duels", type=int, default=100)
    ap.add_argument("--speed-duels", type=int, default=200)
    ap.add_argument("--decks", type=Path, default=Path("../ygo-agent/assets/deck"))
    args = ap.parse_args()
    engine = Engine()
    decks = [load_ydk(p) for p in sorted(args.decks.glob("*.ydk"))]

    def config(s):
        rng = random.Random(s)
        return DuelConfig(rng.choice(decks), rng.choice(decks), seed=s)

    bad, steps = 0, 0
    for s in range(args.duels):
        mismatch, n = lockstep(engine, config(s), s)
        steps += n
        if mismatch:
            bad += 1
            print(f"seed {s}: {mismatch}", flush=True)
    print(f"lockstep: {args.duels} duels, {steps} steps, {bad} mismatching")

    t0, steps, auto = time.time(), 0, 0
    for s in range(10_000, 10_000 + args.speed_duels):
        rng, env = random.Random(s), CppEnv(engine, config(s))
        obs = env.reset()
        while obs is not None:
            obs = env.step(rng.randrange(int(obs["action_mask"].sum())))
        steps += len(env.choices)
        auto += env.decisions - len(env.choices)
    dt = time.time() - t0
    print(
        f"C++ env, 1 thread incl. Python loop + numpy copies: {args.speed_duels} duels in {dt:.1f}s, "
        f"{steps / dt:.0f} agent steps/s ({(steps + auto) / dt:.0f} incl. auto), {dt / args.speed_duels * 1000:.0f} ms/duel"
    )


if __name__ == "__main__":
    main()
