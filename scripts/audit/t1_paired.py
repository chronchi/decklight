"""Paired arm comparison for the turn-1 search test (NOTEBOOK 2026-10-19 "turn-1 search test").

Reads <run>/<arm>/games-*.jsonl, pairs games by game_idx (same deal, same decks, S = seat 0 goes first), and prints
each arm's S win rate on the shared deals, then arm − base: paired difference with a 95% CI (normal approx on the
per-deal differences), W->L / L->W flips and the exact sign-test p. Fails loud on an empty arm or no shared deals.
Usage: uv run python scripts/audit/t1_paired.py runs/<name> [--base raw] [--arms t1h2 t1h1]
"""

import argparse
import json
import math
import sys
from pathlib import Path

from scipy.stats import binomtest


def load(d: Path) -> dict[int, dict]:
    out = {}
    for f in sorted(d.glob("games-*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["game_idx"]] = r
    return out


def score(r: dict) -> float:
    if r.get("truncated") or r["winner"] not in (0, 1):
        return 0.5
    return 1.0 if r["winner"] == r["s_seat"] else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--base", default="raw")
    ap.add_argument("--arms", nargs="+", default=["t1h2", "t1h1"])
    a = ap.parse_args()
    base = load(a.run / a.base)
    if not base:
        sys.exit(f"ALERT t1_paired: no games in {a.run / a.base}")
    print(f"{a.base}: n {len(base)}, S win {sum(map(score, base.values())) / len(base):.3f}")
    for arm in a.arms:
        g = load(a.run / arm)
        if not g:
            print(f"{arm}: no games")
            continue
        ids = sorted(set(g) & set(base))
        if not ids:
            sys.exit(f"ALERT t1_paired: {arm} shares no deals with {a.base}")
        assert all(g[i]["decks"] == base[i]["decks"] and g[i]["seed"] == base[i]["seed"] for i in ids), (
            "deal mismatch"
        )
        d = [score(g[i]) - score(base[i]) for i in ids]
        n = len(d)
        m = sum(d) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, n - 1))
        se = sd / math.sqrt(n)
        wl = sum(1 for x in d if x < 0)
        lw = sum(1 for x in d if x > 0)
        p = binomtest(lw, wl + lw, 0.5).pvalue if wl + lw else 1.0
        sb = sum(score(base[i]) for i in ids) / n
        sa = sum(score(g[i]) for i in ids) / n
        srch = sum(g[i].get("n_searched", 0) for i in ids) / n
        ovr = sum(g[i].get("n_override", 0) for i in ids) / max(
            1, sum(g[i].get("n_searched", 0) for i in ids)
        )
        decks = len({g[i]["decks"][0] for i in ids})
        print(
            f"{arm}: n {n} shared deals ({decks} first-player decks); {a.base} {sb:.3f} -> {arm} {sa:.3f}; "
            f"diff {m * 100:+.1f} pp, 95% CI [{(m - 1.96 * se) * 100:+.1f}, {(m + 1.96 * se) * 100:+.1f}]; "
            f"W->L {wl} / L->W {lw}, sign p {p:.2g}; searches/game {srch:.1f}, override {ovr:.3f}"
        )
        if srch == 0:
            print(f"ALERT t1_paired: {arm} searched nothing")


if __name__ == "__main__":
    main()
