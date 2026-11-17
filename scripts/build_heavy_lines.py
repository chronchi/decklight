#!/usr/bin/env python3
"""The heavy-combo subset of a line-starts file (window PPO, NOTEBOOK 2026-11-08): opponents that build a big turn-1
board or kill on turn 2, closer to the human combo decks P fails to interrupt.

    python3 scripts/build_heavy_lines.py data/cache/line_starts/starts-v2.jsonl \
        --out data/cache/line_starts/starts-v2-heavy.jsonl

Keeps a turn-1 start (`kind` first) when its line's deck has `first.expert_end_board` >= --board, and a turn-2 start
(`kind` second) when the deck's `second_best_of_k.kill` >= --kill (stats from `--pool`, the opponent deck is the
`line_id` prefix). Starts whose deck is not in the pool are dropped and counted. Exit 3 on 0 kept of either kind.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("starts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="config/league/combo_pool_v2.json")
    ap.add_argument("--board", type=float, default=5.0)
    ap.add_argument("--kill", type=float, default=0.25)
    a = ap.parse_args()
    pool = {d["deck"]: d for d in json.loads(Path(a.pool).read_text())["decks"]}
    kept, seen, decks = [], Counter(), Counter()
    for x in Path(a.starts).read_text().splitlines():
        if not x.strip():
            continue
        r = json.loads(x)
        d = pool.get(r["line_id"].split("/")[0])
        seen[r["kind"]] += 1
        if d is None:
            seen["no_pool_stats"] += 1
            continue
        if r["kind"] not in ("first", "second"):
            print(f"ALERT build_heavy_lines: unknown kind {r['kind']!r} ({r['line_id']})", flush=True)
            return 3
        if r["kind"] == "first":
            ok = (d.get("first") or {}).get("expert_end_board", 0) >= a.board
        else:
            ok = (d.get("second_best_of_k") or {}).get("kill", 0) >= a.kill
        if ok:
            kept.append(x)
            decks[(r["kind"], d["deck"])] += 1
    n = Counter(k for k, _ in decks.elements())
    print(json.dumps({"seen": seen, "kept": n, "decks": {f"{k}:{d}": c for (k, d), c in sorted(decks.items())}}),
          flush=True)
    if not n["first"] or not n["second"]:
        print("ALERT build_heavy_lines: 0 kept starts of a kind", flush=True)
        return 3
    Path(a.out).write_text("\n".join(kept) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
