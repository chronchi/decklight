"""Reference for rust/duel-cli: the same seeded random duels through Python's CppEnv, hashed the same way.

    uv run python scripts/app/duel_hash.py --deck0 A.ydk --deck1 B.ydk --games N [--seed S] [--choice-seed C]
    uv run python scripts/app/duel_hash.py ... --compare duel-cli.jsonl   # exit 2 on any mismatch

One JSON line per game (game, seed, decisions, done, winner, hash). The hash is FNV-1a 64 over each observation's
fields in `to_dict` order (masks as uint8, ints little-endian), then the chosen index (4 LE bytes); game g uses duel
seed S+g and a splitmix64 choice stream seeded by C^g (index = next() % num_options). See rust/duel-cli/src/main.rs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FIELDS = ("cards", "card_mask", "global", "chain", "chain_mask", "actions", "action_mask", "history", "history_mask",
          "hist_long", "hist_long_mask", "card_known", "known_code", "priv_code")
M64 = (1 << 64) - 1
FNV_PRIME = 0x100000001B3


def fnv(h: int, b: bytes) -> int:
    for x in b:
        h = ((h ^ x) * FNV_PRIME) & M64
    return h


def obs_bytes(obs: dict) -> bytes:
    out = []
    for k in FIELDS:
        a = obs[k]
        a = a.astype(np.uint8) if a.dtype == bool else a.astype("<i4") if a.dtype.kind == "i" and a.dtype.itemsize == 4 else a
        out.append(np.ascontiguousarray(a).tobytes())
    return b"".join(out)


class SplitMix:
    def __init__(self, s: int) -> None:
        self.s = s & M64

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)


def run(deck0: Path, deck1: Path, games: int, seed: int, cseed: int, max_decisions: int) -> list[dict]:
    from decklight.decks import load_ydk
    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import DuelConfig, Engine

    engine = Engine()
    d0, d1 = load_ydk(deck0), load_ydk(deck1)
    out = []
    for g in range(games):
        s = seed + g
        env = CppEnv(engine, DuelConfig(d0, d1, seed=s))
        rng = SplitMix(cseed ^ g)
        h = 0xCBF29CE484222325
        obs = env.reset()
        steps = 0
        while obs is not None and steps < max_decisions:
            h = fnv(h, obs_bytes(obs))
            n = int(obs["action_mask"].sum())
            idx = rng.next() % n
            h = fnv(h, int(idx).to_bytes(4, "little"))
            obs = env.step(idx)
            steps += 1
        out.append({"game": g, "seed": s, "decisions": steps, "done": bool(env.done), "winner": int(env.winner),
                    "hash": f"{h:016x}"})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck0", type=Path, required=True)
    ap.add_argument("--deck1", type=Path, required=True)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--choice-seed", type=int, default=0)
    ap.add_argument("--max-decisions", type=int, default=3000)
    ap.add_argument("--compare", type=Path, help="duel-cli output (jsonl) to compare against")
    a = ap.parse_args(argv)
    rows = run(a.deck0, a.deck1, a.games, a.seed, a.choice_seed, a.max_decisions)
    for r in rows:
        print(json.dumps(r))
    if a.compare:
        other = [json.loads(line) for line in a.compare.read_text().splitlines() if line.startswith("{")]
        keys = ("game", "seed", "decisions", "done", "winner", "hash")
        bad = [(x, y) for x, y in zip(rows, other, strict=False) if any(x[k] != y[k] for k in keys)]
        if len(other) != len(rows) or bad:
            print(f"ALERT duel_hash: {len(bad)} of {len(rows)} games differ (other has {len(other)} rows); "
                  f"first: {bad[:1]}", file=sys.stderr)
            return 2
        print(f"OK: {len(rows)} games identical", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
