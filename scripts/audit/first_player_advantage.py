"""Going-first advantage in the primer games (data/primers/summaries.json.gz).

Overall first-player win rate with a Wilson CI and an exact binomial test vs 0.5, split by game source, plus a
per-deck table: win rate going first vs going second (Fisher exact test per deck, Holm-corrected across decks).
Draws are dropped. Usage: uv run python scripts/audit/first_player_advantage.py [--min-games 20]
"""

import argparse
import gzip
import json
from collections import defaultdict

from scipy.stats import binomtest, fisher_exact


def source(seed: int) -> str:
    if 10000 <= seed < 20000:
        return "pool-v2 (raw P, new deal each)"
    if 20000 <= seed < 30000:
        return "newcards-v2 (raw P)"
    return "hppo (Full search games)"


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", default="data/primers/summaries.json.gz")
    ap.add_argument("--min-games", type=int, default=20)
    a = ap.parse_args()
    games = json.load(gzip.open(a.summaries))
    by_src = defaultdict(lambda: [0, 0])
    deck = defaultdict(lambda: [0, 0, 0, 0])  # first wins, first n, second wins, second n
    draws = 0
    for g in games:
        if g["winner"] not in (0, 1):
            draws += 1
            continue
        fp = g["first_player"]
        fw = int(g["winner"] == fp)
        s = by_src[source(g["seed"])]
        s[0] += fw
        s[1] += 1
        for seat, name in enumerate(g["decks"]):
            won = int(g["winner"] == seat)
            r = deck[name]
            if seat == fp:
                r[0] += won
                r[1] += 1
            else:
                r[2] += won
                r[3] += 1
    assert sum(n for _, n in by_src.values()) > 0, "no decided games"
    k = sum(w for w, _ in by_src.values())
    n = sum(n for _, n in by_src.values())
    print(f"games {len(games)}, draws dropped {draws}")
    for name, (w, m) in [*sorted(by_src.items()), ("ALL", (k, n))]:
        lo, hi = wilson(w, m)
        p = binomtest(w, m, 0.5).pvalue
        print(f"{name:34s} first wins {w}/{m} = {w / m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  p={p:.2g}")
    rows = []
    for name, (fw, fn, sw, sn) in deck.items():
        if fn >= a.min_games and sn >= a.min_games:
            p = fisher_exact([[fw, fn - fw], [sw, sn - sw]]).pvalue
            rows.append((fw / fn - sw / sn, name, fw, fn, sw, sn, p))
    rows.sort(reverse=True)
    adj = holm([r[-1] for r in rows])
    pos = sum(r[0] > 0 for r in rows)
    print(f"\ndecks with >= {a.min_games} games each way: {len(rows)}; first > second in {pos} "
          f"(sign test p={binomtest(pos, len(rows), 0.5).pvalue:.2g}); Holm p<0.05 in {sum(q < 0.05 for q in adj)}")
    print(f"{'deck':28s} {'1st':>12s} {'2nd':>12s} {'diff':>6s} {'holm p':>7s}")
    for (dlt, name, fw, fn, sw, sn, _), q in [*zip(rows[:12], adj[:12]), *zip(rows[-8:], adj[-8:])]:
        print(f"{name[:28]:28s} {fw:3d}/{fn:3d} {fw / fn:.2f} {sw:3d}/{sn:3d} {sw / sn:.2f} {dlt:+.2f} {q:7.2g}")


if __name__ == "__main__":
    main()
