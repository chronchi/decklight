"""Paired comparison of two search_match arms on the same deals (games-*.jsonl of vast_search_match.sh).

Pairs games by (decks, seed, s_seat) and scores B minus A (win 1, loss 0, truncated 0.5) overall and per seat of the
scored side (s_seat 0 = went first). Prints n, the paired difference with a 95% CI, and the strict flip counts.
Exit 2 when fewer than 90% of A's games pair up. Engine-free.

    python scripts/audit/paired_sm.py runs/stage-b1-20261013-0008/ftswv runs/stage-b1-20261013-0008/bclean
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def games(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("games-*.jsonl")):
        for raw in f.read_text().splitlines():
            if raw.strip():
                g = json.loads(raw)
                score = 0.5 if g.get("truncated") else float(g["winner"] == g["s_seat"])
                out[(tuple(g["decks"]), g["seed"], g["s_seat"])] = score
    return out


def report(name: str, diffs: list[float]) -> str:
    n = len(diffs)
    if n < 2:
        return f"{name}: n={n}"
    m = sum(diffs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1) / n)
    up, down = sum(x > 0 for x in diffs), sum(x < 0 for x in diffs)
    return f"{name}: n={n} B-A {m * 100:+.1f} pp [{(m - 1.96 * se) * 100:+.1f}, {(m + 1.96 * se) * 100:+.1f}] (SE {se * 100:.2f}); B better {up}, A better {down}"


def main() -> int:
    a, b = games(Path(sys.argv[1])), games(Path(sys.argv[2]))
    keys = sorted(set(a) & set(b))
    if len(keys) < 0.9 * len(a):
        print(f"ALERT paired_sm: only {len(keys)} of {len(a)} games pair up")
        return 2
    print(f"A={sys.argv[1]} ({sum(a.values()) / len(a):.3f}, n={len(a)})  B={sys.argv[2]} ({sum(b.values()) / len(b):.3f}, n={len(b)})")
    print(report("all", [b[k] - a[k] for k in keys]))
    for seat, label in ((0, "first"), (1, "second")):
        print(report(label, [b[k] - a[k] for k in keys if k[2] == seat]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
