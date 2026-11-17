"""Step-2 opponent admit list from the step-1a measurements (docs/plans/2026-11-03-opp-turn-interaction.md; NOTEBOOK
2026-11-03 22:30 for why the rules are these).

Inputs: combo_pool.py rows (one or more JSON files) and rank_lines.py output. Rules:
- `admit_turn1_lines`: the deck has a combo_pool row whose going-first test replayed, >= MIN_HANDS opening hands with
  verified lines, and the best line per hand averages >= MIN_ACTS player-0 activations / summons (windows for us).
- `going_second_candidate`: decks to generate going-second lines for (step 1b). P-driven numbers only rank them
  (they are the noisy signal the plan avoids for training): per-sample kill >= 25% or P kill >= 25%.
Decks with lines but no combo_pool row are listed as `unmeasured` and not admitted.

    uv run python scripts/audit/combo_admit.py runs/combo-pool-X/s*.json --lines data/cache/combo_lines/ranked.jsonl \
        --out config/league/combo_pool_v2.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MIN_HANDS, MIN_ACTS, MIN_KILL = 5, 4.0, 0.25


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+", type=Path)
    ap.add_argument("--lines", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = {r["deck"]: r for f in a.rows for r in json.loads(f.read_text())["rows"]}
    best = defaultdict(list)
    for raw in a.lines.read_text().splitlines():
        r = json.loads(raw)
        if r["best_for_hand"]:
            best[r["deck"]].append(r)
    if not rows or not best:
        print(f"ALERT combo_admit: {len(rows)} combo_pool rows, {len(best)} ranked decks", flush=True)
        return 3
    out, unmeasured = [], sorted(set(best) - set(rows))
    for d in sorted(rows):
        r, b = rows[d], best.get(d, [])
        if "error" in r:
            out.append({"deck": d, "error": r["error"], "admit_turn1_lines": False, "going_second_candidate": False})
            continue
        acts = round(statistics.mean(x["acts"] for x in b), 2) if b else None
        f_int = round(statistics.mean(x["f_int"] for x in b), 2) if b else None
        first = r.get("first") or {}
        s2 = r["second_p"]
        lines = bool(first.get("hands")) and len(b) >= MIN_HANDS
        t1 = lines and acts >= MIN_ACTS
        g2 = r.get("second_sample_kill", 0) >= MIN_KILL or s2["kill"] >= MIN_KILL
        row = {"deck": d, "hands_with_lines": len(b), "best_line_acts": acts,
               "best_line_field_interruptions": f_int, "first": first or None, "second_p": s2,
               "second_sample_kill": r.get("second_sample_kill"), "second_best_of_k": r.get("second_best_of_k"),
               "alerts": r.get("alerts", [])}  # fmt: skip
        out.append({**row, "admit_turn1_lines": t1, "going_second_candidate": g2})
    t1n, g2n = sum(o["admit_turn1_lines"] for o in out), sum(o["going_second_candidate"] for o in out)
    if t1n == 0:
        print("ALERT combo_admit: no deck admitted for turn-1 lines", flush=True)
        return 3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rules = {"admit_turn1_lines": f"combo_pool going-first replayed, >= {MIN_HANDS} hands with lines, best-line acts "
                                  f">= {MIN_ACTS}",
             "going_second_candidate": f"per-sample or P kill >= {MIN_KILL:.0%} going second"}  # fmt: skip
    rules["going_second_candidate"] += " (candidates for step-1b line generation, not opponents)"
    a.out.write_text(json.dumps({"made": "2026-11-03", "inputs": [str(p) for p in a.rows] + [str(a.lines)],
                                 "rules": rules, "unmeasured": unmeasured, "decks": out}, indent=1))  # fmt: skip
    print(f"{len(out)} decks: {t1n} turn-1 line opponents, {g2n} going-second candidates, {len(unmeasured)} unmeasured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
