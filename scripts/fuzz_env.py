"""Fuzz the Python reference env with uniformly random option choices; print coverage and limits.

uv run python scripts/fuzz_env.py --duels 300 [--decks ../ygo-agent/assets/deck]
"""

import argparse
import collections
import random
import time
from pathlib import Path

from decklight.decks import load_ydk
from decklight.engine.actions import Kind
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duels", type=int, default=100)
    ap.add_argument("--decks", type=Path, default=Path("../ygo-agent/assets/deck"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    engine = Engine()
    decks = [load_ydk(p) for p in sorted(args.decks.glob("*.ydk"))]
    kinds, msgs = collections.Counter(), collections.Counter()
    steps = auto = truncated = max_rows = max_opts = 0
    t0 = time.time()
    for s in range(args.seed, args.seed + args.duels):
        rng = random.Random(s)
        env = PyEnv(engine, DuelConfig(rng.choice(decks), rng.choice(decks), seed=s))
        obs = env.reset()
        while obs is not None:
            n = int(obs["action_mask"].sum())
            kinds.update(Kind(int(k)).name for k in obs["actions"][:n, 0])
            msgs[int(obs["global"][21])] += 1
            max_rows = max(max_rows, int(obs["card_mask"].sum()) + int(obs["global"][22]))
            max_opts = max(max_opts, n)
            obs = env.step(rng.randrange(n))
        steps += len(env.choices)
        auto += env.decisions - len(env.choices)
        truncated += env.truncated
    dt = time.time() - t0
    print(f"{args.duels} duels, {steps} agent steps, {auto} auto steps, {dt:.0f}s ({steps / dt:.0f} steps/s)")
    print(f"max card rows {max_rows}, max options {max_opts}, truncated decisions {truncated}")
    print("option kinds:", dict(sorted(kinds.items())))
    print("decision msgs:", dict(sorted(msgs.items())))


if __name__ == "__main__":
    main()
