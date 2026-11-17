#!/usr/bin/env python3
"""Split a playtest report's match by whether our deck (deck A) is a holdout deck (NOTEBOOK 2026-11-08 20:30 rule 2).

    python3 scripts/audit/card_split.py runs/<card>/<arm>/playtest/report.json [--match base]

Per group: games, win rate, 95% CI from deck-pair clusters (games of one ordered deck pair are not independent),
and the first-seat win rate. Exit 3: no game in a group, or a deck name not in the report's deck sample.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict


def summarize(games: list) -> dict:
    by = defaultdict(list)
    for g in games:
        by[(g[0], g[1])].append(g[4])
    n, w = len(games), sum(g[4] for g in games)
    p = w / n
    # cluster-robust variance of the mean over deck pairs
    var = sum((sum(v) - p * len(v)) ** 2 for v in by.values()) / n**2 * len(by) / max(len(by) - 1, 1)
    se = math.sqrt(var)
    first = [g[4] for g in games if g[3]]
    return {"games": n, "win_rate": round(p, 3), "ci95": [round(p - 1.96 * se, 3), round(p + 1.96 * se, 3)],
            "pairs": len(by), "first": round(sum(first) / max(len(first), 1), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report")
    ap.add_argument("--match", default="base")
    a = ap.parse_args()
    r = json.load(open(a.report))
    hold, sample = set(r["holdout"]), set(r["decks"])
    games = r["matches"][a.match]["per_game"]
    if any(g[0] not in sample or g[1] not in sample for g in games):
        print("ALERT card_split: a deck name is not in the report's sample", flush=True)
        return 3
    out = {}
    for name, sel in (("all", lambda g: True), ("ours_holdout", lambda g: g[0] in hold),
                      ("ours_train", lambda g: g[0] not in hold), ("both_holdout", lambda g: g[0] in hold and g[1] in hold)):
        gs = [g for g in games if sel(g)]
        if not gs:
            print(f"ALERT card_split: no game in {name}", flush=True)
            return 3
        out[name] = summarize(gs)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
