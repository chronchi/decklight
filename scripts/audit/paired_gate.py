"""Paired ygo-agent gate: did checkpoint B really beat checkpoint A on the same deals?

Pairs `per_game` rows of two playtest report.json files by (deck_a, deck_b, seed,
a_first) and scores the per-pair difference (A's win minus B's win, each in
{1, 0.5, 0}): the pairing removes deck/seat/seed variance, so small but real
gains survive the ±6 pts unpaired noise of a single 248-game card.

Usage: python scripts/audit/paired_gate.py <report_A> <report_B> [--match ygoagent]
Prints: n pairs, W->L / L->W / ties (strict 1-vs-0 flips; half-point pairs listed
separately), the exact two-sided sign-test p-value on the strict flips, the paired
win-rate difference with a 95% CI (normal approx on the per-pair differences), and
the same split by seat (A first / second). Exit 2 when fewer than 90% of rows
pair up (the schedules differ) or a match section is missing. Engine-free: JSON only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def rows_of(path: Path, match: str) -> list:
    if path.is_dir():  # a search_match --opp ygoagent output dir: games-*.jsonl rows, S = A
        recs = [json.loads(x) for f in sorted(path.glob("games-*.jsonl"))
                for x in f.read_text().splitlines() if x.strip()]
        return [[g["deck_s"], g["deck_r"], g["seed"], g["s_first"], g["win"], g["turns"], g["cut"]]
                for g in recs]
    r = json.loads(path.read_text())
    try:
        return r["matches"][match]["per_game"]
    except KeyError:
        return []


def key_of(row) -> tuple:
    return (row[0], row[1], row[2], bool(row[3]))


def signtest_p(wl: int, lw: int) -> float:
    """Exact two-sided binomial p-value for `wl` vs `lw` strict flips (H0: p = 0.5)."""
    n = wl + lw
    if n == 0:
        return 1.0
    m = min(wl, lw)
    tail = sum(math.comb(n, i) for i in range(m + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def diff_ci(diffs: list[float]) -> tuple[float, float, float]:
    """Mean paired difference with a 95% CI (normal approx on per-pair differences)."""
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return mean, mean, mean
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    h = 1.96 * math.sqrt(var / n)
    return mean, mean - h, mean + h


def section(title: str, pairs: list[tuple[float, float]]) -> None:
    n = len(pairs)
    wl = sum(1 for a, b in pairs if a == 1.0 and b == 0.0)
    lw = sum(1 for a, b in pairs if a == 0.0 and b == 1.0)
    ties = sum(1 for a, b in pairs if a == b)
    half = n - wl - lw - ties
    p = signtest_p(wl, lw)
    mean, lo, hi = diff_ci([a - b for a, b in pairs])
    wra = sum(a for a, _ in pairs) / n
    wrb = sum(b for _, b in pairs) / n
    print(f"{title:<14} n={n:4d}  A {wra:.3f} vs B {wrb:.3f}  diff {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"{'':14} W->L {wl:4d}  L->W {lw:4d}  ties {ties:4d}  half {half:4d}  sign-test p={p:.5g}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report_a", type=Path)
    ap.add_argument("report_b", type=Path)
    ap.add_argument("--match", default="ygoagent")
    args = ap.parse_args(argv)

    ra, rb = rows_of(args.report_a, args.match), rows_of(args.report_b, args.match)
    if not ra or not rb:
        print(f"paired_gate: match {args.match!r} missing or empty in "
              f"{args.report_a} (n={len(ra)}) or {args.report_b} (n={len(rb)})", flush=True)
        return 2
    da = {key_of(r): float(r[4]) for r in ra}
    db = {key_of(r): float(r[4]) for r in rb}
    keys = sorted(set(da) & set(db))
    cover = len(keys) / max(len(da), len(db))
    print(f"paired_gate: match={args.match} A={args.report_a} B={args.report_b}")
    print(f"pairs {len(keys)} / A {len(da)} / B {len(db)} (coverage {cover:.1%})")
    if cover < 0.9:
        print("paired_gate: fewer than 90% of rows pair up (schedules differ)", flush=True)
        return 2
    pairs = [(da[k], db[k]) for k in keys]
    section("overall", pairs)
    section("A first", [(da[k], db[k]) for k in keys if k[3]])
    section("A second", [(da[k], db[k]) for k in keys if not k[3]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
