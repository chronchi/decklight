#!/usr/bin/env python3
"""Split a `line_eval.py` compare output by line kind (window PPO, NOTEBOOK 2026-11-08): `first` = the opponent goes
first with a turn-1 combo; `second` = we went first and set up, the opponent plays its turn-2 line.

    python3 scripts/audit/line_eval_split.py runs/<job>/line-eval.json [...] [--out split.json]

Per kind and arm (A = --ckpt, B = --vs-ckpt): win rate, paired B - A with its SE, share of games where we activated
on the line's turn, activations per game by location (set_st = set spell/trap), the opponent's line completed
(not interrupted), scripted decisions before our first activation (median), LP lost and the opponent's field at the
end of the turn. Several files (shards) are pooled. Exit 3 on 0 finished pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def arm(gs: list[dict]) -> dict:
    n = len(gs)
    locs = sorted({k for g in gs for k in g.get("acts_loc", {})})
    first = sorted(g["first_act_followed"] for g in gs if g.get("first_act_followed") is not None)
    m = lambda f: round(sum(f(g) for g in gs) / n, 3) if n else None  # noqa: E731
    return {"n": n, "won": m(lambda g: g["won"] or 0.0), "activated": m(lambda g: g["acts_line_turn"] > 0),
            "acts": m(lambda g: g["acts_line_turn"]),
            **{f"acts_{k}": m(lambda g, k=k: g.get("acts_loc", {}).get(k, 0)) for k in locs},
            "line_completed": m(lambda g: g["completed"]), "broken": m(lambda g: g["broken_at"] is not None),
            "first_act_followed_median": first[len(first) // 2] if first else None,
            "lp_lost": m(lambda g: g["lp_lost_line_turn"]), "opp_field_end": m(lambda g: g["opp_field_end_turn"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    games = [g for f in a.files for g in json.loads(f.read_text())["games"]]
    out, rc = {}, 0
    for kind in ("first", "second", "all"):
        gs = [g for g in games if kind == "all" or g["A"]["kind"] == kind]
        gs = [g for g in gs if g["A"]["won"] is not None and g["B"]["won"] is not None]
        if not gs:
            continue
        d = [g["B"]["won"] - g["A"]["won"] for g in gs]
        se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else None
        out[kind] = {"pairs": len(gs), "B_minus_A": round(statistics.mean(d), 4), "se": round(se, 4) if se else None,
                     "B_better": sum(x > 0 for x in d), "B_worse": sum(x < 0 for x in d),
                     "A": arm([g["A"] for g in gs]), "B": arm([g["B"] for g in gs])}
        print(json.dumps({kind: out[kind]}), flush=True)
    if "all" not in out:
        print("ALERT line_eval_split: 0 finished pairs", flush=True)
        rc = 3
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
