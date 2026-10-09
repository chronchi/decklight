"""Fail fast if search_match's games do not reproduce the report card's (NOTEBOOK 2026-10-09 08:40).

    python3 scripts/audit/harness_check.py <search_match variant dir> <reference.jsonl> [--min 0.99]

Reference rows: {"decks": [seat0, seat1], "seed", "choices"} (card games of the same policy on the same deals, e.g.
data/cache/harness/card-base_r3-744.jsonl from runs/rl5ga-20261008-2015/it0). Pairs by (decks, seed); exits 1 when the
move-identical share is below --min or fewer than 90% of the reference deals are present. JSON only, no engine.
"""

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("ref", type=Path)
ap.add_argument("--min", type=float, default=0.99)
a = ap.parse_args()
ref = {}
for x in a.ref.read_text().splitlines():
    r = json.loads(x)
    ref[(*r["decks"], r["seed"])] = r["choices"]
got = {}
for f in sorted(a.run.glob("games-*.jsonl")):
    for x in f.read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            got[(*r["decks"], r["seed"])] = r["choices"]
common = sorted(set(ref) & set(got))
same = sum(ref[k] == got[k] for k in common)
share = same / len(common) if common else 0.0
print(f"harness_check: {len(common)}/{len(ref)} reference deals present, move-identical {same} ({share:.3f})")
if len(common) < 0.9 * len(ref) or share < a.min:
    print(f"ALERT harness_check: identical share {share:.3f} < {a.min} or too few deals ({len(common)})")
    sys.exit(1)
