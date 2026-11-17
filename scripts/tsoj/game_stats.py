#!/usr/bin/env python3
"""One line per arbiter records file: games, bad results (bot_error/illegal), median turns and decisions.

    python3 scripts/tsoj/game_stats.py RECORDS.jsonl [--min-decisions 30] [--max-bad 0.02]

Exit 3 when the games cannot be real duels (median decisions below --min-decisions) or the bad share is above
--max-bad (NOTEBOOK 2026-11-07: the first box run played 616 games in 157 s)."""
import argparse, json, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("records")
ap.add_argument("--min-decisions", type=float, default=30)
ap.add_argument("--max-bad", type=float, default=0.02)
a = ap.parse_args()
rows = [json.loads(x) for x in open(a.records) if x.strip()]
if not rows:
    print(f"{a.records}: 0 games"); sys.exit(3)
res = [r.get("result") or {} for r in rows]
bad = [x for x in res if isinstance(x.get("reason"), str) and x["reason"].startswith(("bot_error", "illegal"))]
dec = statistics.median(len(r.get("responses", [])) for r in rows)
turns = statistics.median(int(x.get("turns") or 0) for x in res)
reasons = {}
for x in res:
    reasons[str(x.get("reason"))] = reasons.get(str(x.get("reason")), 0) + 1
print(f"{a.records}: {len(rows)} games, bad {len(bad)}, median responses {dec}, median turns {turns}, reasons {reasons}")
sys.exit(3 if dec < a.min_decisions or len(bad) > a.max_bad * len(rows) else 0)
