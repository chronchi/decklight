"""Separate deck strength from piloting skill in policy-vs-ygo-agent card games (both sides use ygo-agent's decks).
Model: P(we win) = sigmoid(c + first*f + d[our deck] + g[our deck] - d[their deck]).
  d[X]: deck strength (shared by both pilots); g[X]: our piloting gap on X relative to ygo-agent (mean-zero);
  c: our overall gap; f: going-first effect for us. Ring pairing (each deck is A vs its 4 successors, B vs its 4
  predecessors) identifies d from the B role and d+g from the A role. L2 on d and g; plain gradient ascent.
Usage: python scripts/audit/pilot_gap.py <report.json>...  (laptop-safe, numpy only)"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    rows = []
    for rp in sys.argv[1:]:
        r = json.loads(Path(rp).read_text())
        rows += r.get("matches", {}).get("ygoagent", {}).get("per_game", [])
    rows = [x for x in rows if x[4] in (0.0, 1.0)]
    decks = sorted({x[0] for x in rows} | {x[1] for x in rows})
    ix = {d: i for i, d in enumerate(decks)}
    a = np.array([ix[x[0]] for x in rows]); b = np.array([ix[x[1]] for x in rows])
    first = np.array([1.0 if x[3] else -1.0 for x in rows]); y = np.array([x[4] for x in rows])
    n = len(decks)
    d, g, c, f = np.zeros(n), np.zeros(n), 0.0, 0.0
    lam, lr = 2.0, 0.5
    for _ in range(4000):
        z = c + f * first + d[a] + g[a] - d[b]
        e = y - 1 / (1 + np.exp(-z))
        gd = np.bincount(a, e, n) - np.bincount(b, e, n) - lam * d
        gg = np.bincount(a, e, n) - lam * g
        d += lr * gd / len(y) * n / 4
        g += lr * gg / len(y) * n / 4
        g -= g.mean(); d -= d.mean()
        c += lr * e.mean(); f += lr * (e * first).mean()
    print(f"{len(y)} decisive games, {n} decks; overall gap c = {c:+.2f} logits "
          f"(P = {1 / (1 + np.exp(-c)):.3f} on an even deck), first-player effect for us f = {f:+.2f}")
    print(f"\n{'deck':<18} {'strength d':>10} {'our pilot gap g':>16}")
    for i in np.argsort(g):
        print(f"{decks[i]:<18} {d[i]:+10.2f} {g[i]:+16.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
