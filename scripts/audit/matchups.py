"""Per-deck / per-matchup breakdown of report-card games (playtest report.json `per_game`), pooled over cards.

Each per_game row: [deck_a (ours), deck_b (opponent), seed, a_first, win (1/0.5/0), turns, cut].
Usage: python scripts/audit/matchups.py <match> <report.json>... [--label NAME=report.json ...]
  match: ygoagent | base | greedy | new_base | ...
Prints, pooled over all given reports: win rate by our deck, by opponent deck, by who goes first, and the turn
distribution of wins and losses; then per-report totals so checkpoints can be compared on identical schedules.
Laptop-safe: reads JSON only.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def rows_of(path: Path, match: str) -> list:
    r = json.loads(path.read_text())
    return r.get("matches", {}).get(match, {}).get("per_game", [])


def table(title: str, groups: dict, min_n: int = 1) -> None:
    print(f"\n{title}")
    items = sorted(((sum(v) / len(v), k, len(v)) for k, v in groups.items() if len(v) >= min_n), reverse=True)
    for wr, k, n in items:
        print(f"  {k:<44} {wr:5.3f}  n={n}")


def main() -> int:
    match, paths = sys.argv[1], [Path(p) for p in sys.argv[2:]]
    allrows = []
    for p in paths:
        rows = rows_of(p, match)
        if not rows:
            continue
        allrows += rows
        w = sum(x[4] for x in rows)
        first = [x[4] for x in rows if x[3]]
        second = [x[4] for x in rows if not x[3]]
        print(f"{p.parent.parent.name if p.parent.name == 'playtest' else p}: {w / len(rows):.3f} n={len(rows)}  "
              f"first {sum(first) / max(1, len(first)):.3f}  second {sum(second) / max(1, len(second)):.3f}")
    if not allrows:
        print("no games", file=sys.stderr)
        return 2
    ours, theirs, pair, first = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for a, b, _s, a_first, win, _turns, _cut in allrows:
        ours[a].append(win)
        theirs[b].append(win)
        pair[f"{a} vs {b}"].append(win)
        first["we go first" if a_first else "we go second"].append(win)
    print(f"\npooled: {sum(x[4] for x in allrows) / len(allrows):.3f} over {len(allrows)} games, {len(paths)} reports")
    table("by who goes first", first)
    table("by OUR deck (our win rate when we pilot it)", ours)
    table("by OPPONENT deck (our win rate against it)", theirs)
    wins = sorted(x[5] for x in allrows if x[4] == 1.0)
    losses = sorted(x[5] for x in allrows if x[4] == 0.0)

    def q(v: list) -> str:
        return " ".join(str(v[int(len(v) * f)]) for f in (0.1, 0.25, 0.5, 0.75, 0.9)) if v else "-"

    print(f"\nturns (p10 p25 p50 p75 p90): wins {q(wins)} | losses {q(losses)}")
    fast = [x for x in allrows if x[4] == 0.0 and x[5] <= 3]
    print(f"losses by turn <= 3: {len(fast)} / {len(losses)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
