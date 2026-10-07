"""How familiar is each ygo-agent gauntlet deck to our BC policy? Card overlap with decks/pool (the BC deck pool)
and with the decks that have verified demo lines, next to our pooled win rate when piloting the deck vs ygo-agent.
Usage: python scripts/audit/gauntlet_coverage.py <report.json>...  (laptop-safe: ydk + json only)"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ids(p: Path) -> list[int]:
    return [int(x) for x in p.read_text().split() if x.isdigit()]


def main() -> int:
    pool = set()
    for p in (ROOT / "decks/pool").glob("*.ydk"):
        pool |= set(ids(p))
    demo_decks = {p.parent.parent.name for p in (ROOT / "data/demos").glob("*/verified/*.jsonl")}
    demo = set()
    for name in demo_decks:
        p = ROOT / "decks/pool" / f"{name}.ydk"
        if p.is_file():
            demo |= set(ids(p))
    ours = defaultdict(list)
    for rp in sys.argv[1:]:
        r = json.loads(Path(rp).read_text())
        for a, _b, _s, _f, win, _t, _c in r.get("matches", {}).get("ygoagent", {}).get("per_game", []):
            ours[a].append(win)
    rows = []
    for p in sorted((ROOT / "decks/gauntlet").glob("*.ydk")):
        main_ids = ids(p)
        u = set(main_ids)
        cov_pool = sum(c in pool for c in main_ids) / len(main_ids)
        cov_demo = sum(c in demo for c in main_ids) / len(main_ids)
        w = ours.get(p.stem, [])
        rows.append((sum(w) / len(w) if w else float("nan"), p.stem, cov_pool, cov_demo, len(u)))
    print(f"{'deck':<18} {'our wr':>6} {'cards in pool':>13} {'in demo decks':>13}")
    for wr, name, cp, cd, _n in sorted(rows, reverse=True):
        print(f"{name:<18} {wr:6.3f} {cp:13.2f} {cd:13.2f}")
    xs = [(r[2], r[0]) for r in rows if r[0] == r[0]]
    n = len(xs)
    mx, my = sum(x for x, _ in xs) / n, sum(y for _, y in xs) / n
    cov = sum((x - mx) * (y - my) for x, y in xs)
    vx, vy = sum((x - mx) ** 2 for x, _ in xs), sum((y - my) ** 2 for _, y in xs)
    print(f"\ncorrelation(pool coverage, our wr) = {cov / (vx * vy) ** 0.5:.2f} over {n} decks; demo decks: {len(demo_decks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
