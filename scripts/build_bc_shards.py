"""Replay every demo record source into BC shards (see ygo_zero.model.bc_data).

uv run python scripts/build_bc_shards.py --out data/cache/bc_shards --jobs 2
"""

import argparse
from pathlib import Path

from ygo_zero.model.bc_data import build_shards

SOURCES = {  # tag -> globs of record files (verified lines and their hand-variant expansions)
    "guides": ["data/demos/verified/*.jsonl", "data/demos/expanded/*.jsonl"],
    "claude": ["data/demos/claude/verified/*.jsonl", "data/demos/claude/expanded*/*.jsonl"],
    "muse": ["data/demos/planned*/verified/*.jsonl", "data/demos/planned*/expanded*/*.jsonl"],
    "games": ["data/demos/games/*.jsonl"],  # ADR 0004 L2-L4 (deck label = actor's deck)
}

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("data/cache/bc_shards"))
ap.add_argument("--jobs", type=int, default=2)
args = ap.parse_args()
src = {
    tag: sorted(f for g in globs for f in Path(".").glob(g) if f.stat().st_size)
    for tag, globs in SOURCES.items()
}
print({k: len(v) for k, v in src.items()}, "files")
print("samples:", build_shards(src, args.out, args.jobs))
