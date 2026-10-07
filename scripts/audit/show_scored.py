"""Read play-out-scored turn-1 lines (scripts/score_lines.py output): per hand the greedy line, the best sample by end
field and the best by rollout score, rendered with show_lines.render, plus the rollout score and its samples.

    python scripts/audit/show_scored.py <scored.jsonl> [--deck D] [--hand H] [--source S] [--disagree] [--max-hands N]
--disagree keeps only hands where the rollout winner has less end field than the field winner (the cases the new
score is meant to fix: e.g. hand traps kept instead of discarded for board). Laptop-safe: engine replay only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import show_lines as SL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scored", type=Path)
    ap.add_argument("--deck")
    ap.add_argument("--hand", type=int)
    ap.add_argument("--source")
    ap.add_argument("--disagree", action="store_true")
    ap.add_argument("--max-hands", type=int, default=3)
    a = ap.parse_args()
    g = defaultdict(list)
    for x in a.scored.read_text().splitlines():
        r = json.loads(x)
        g[(r["deck"], r["source"], r["hand"])].append(r)
    nn, engine = SL.CardNames().names, SL.Engine()
    shown = 0
    for (deck, src, h), rs in sorted(g.items()):
        if (a.deck and deck != a.deck) or (a.hand is not None and h != a.hand) or (a.source and src != a.source):
            continue
        gr = next((r for r in rs if r["kind"] == "greedy"), None)
        sm = [r for r in rs if r["kind"] != "greedy"]
        if not (gr and sm):
            continue
        br = max(sm, key=lambda r: r["rollout"])
        bf = max(sm, key=lambda r: (r["field"], r["v_p0"]))
        if a.disagree and not (br is not bf and br["field"] < bf["field"]):
            continue
        if shown >= a.max_hands:
            break
        shown += 1
        print(f"== {deck} {src} hand {h}")
        for name, r in (("greedy", gr), ("best_field", bf), ("best_rollout", br)):
            if name == "best_field" and r is br:
                continue
            ss = " ".join(f"{s:+.1f}" for s in r["rollout_samples"])
            print(f"  -- {name}: rollout {r['rollout']:+.2f} done {r['rollout_done']:+.2f} [{ss}]")
            print("\n".join(SL.render(r, engine, nn)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
