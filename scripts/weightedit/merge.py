#!/usr/bin/env python3
"""Merge play.py shard summaries (--shard i/n) into one: wins and games add up; rates and means are averaged
weighted by games (approximate for per-window rates). Prints the merged JSON; exit 3 if a shard has 0 games.

    python3 scripts/weightedit/merge.py runs/weightedit-20261109/<name>.holdout.s*.json --out runs/weightedit-20261109/<name>.holdout.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

KEYS = ("win_rate", "interrupt_rate", "battle_rate", "attack_rate", "interrupt_turn_rate", "damage_per_turn",
        "opp_field_removed", "opp_board_left", "first_use_window", "first_use_frac")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    parts = [json.loads(p.read_text()) for p in a.shards]
    if any(not d["A"]["games"] for d in parts):
        print("ALERT merge: a shard has 0 games", flush=True)
        return 3
    out = {"a": parts[0]["a"], "b": parts[0]["b"], "holdout": parts[0]["holdout"], "shards": len(parts)}
    for side in ("A", "B"):
        g = [d[side]["games"] for d in parts]
        m = {"games": sum(g)}
        for k in KEYS:
            vals = [(d[side].get(k), n) for d, n in zip(parts, g) if d[side].get(k) is not None]
            m[k] = round(sum(v * n for v, n in vals) / sum(n for _, n in vals), 3) if vals else None
        out[side] = m
    n, p = out["A"]["games"], out["A"]["win_rate"]
    z, d = 1.96, 1 + 1.96**2 / n  # Wilson interval
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    out["A"]["win_ci"] = [round(c - h, 3), round(c + h, 3)]
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
