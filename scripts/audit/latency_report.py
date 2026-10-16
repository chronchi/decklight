"""Per-move latency of a --latency 1 search_match arm.

    python scripts/audit/latency_report.py DIR

Prints p50/p95/max of lat_raw (raw moves), lat_search (searched moves) and lat_turn (own turns) over all games in
DIR/games-*.jsonl. Exits 1 when no game carries latency data (the arm was not run with --latency 1).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="search_match --out dir (games-*.jsonl with lat_raw/lat_search/lat_turn)")
    a = ap.parse_args(argv)
    raw, search, turn = [], [], []
    n_games = 0
    for f in sorted(glob.glob(f"{a.dir}/games-*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                n_games += 1
                raw.extend(float(x) for x in r.get("lat_raw", []))
                search.extend(float(x) for x in r.get("lat_search", []))
                turn.extend(float(x) for x in r.get("lat_turn", []))
    if not raw and not search and not turn:
        print(f"ALERT latency_report: no latency data in {a.dir} (run search_match with --latency 1)")
        return 1
    print(f"latency over {n_games} games (seconds):")
    print(f"  {'':10} {'n':>7} {'p50':>8} {'p95':>8} {'max':>8}")
    for name, xs in (("lat_raw", raw), ("lat_search", search), ("lat_turn", turn)):
        if xs:
            print(f"  {name:10} {len(xs):>7} {pct(xs, 0.5):>8.3f} {pct(xs, 0.95):>8.3f} {max(xs):>8.3f}")
        else:
            print(f"  {name:10} {0:>7} {'-':>8} {'-':>8} {'-':>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
