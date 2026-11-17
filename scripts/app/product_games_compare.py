"""Compare the Python and Rust A3b-1 product-game dumps; exit 2 on any difference (wall times excluded).

    uv run python scripts/app/product_games_compare.py data/cache/app/product_py.jsonl.gz \\
        data/cache/app/product_rs.jsonl.gz

Wall-time fields (`wall_bits`, `lat_*_bits`, `game_wall_bits`) are excluded; every other field, including f64 score
bit patterns and rounded sample bits, must be identical.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys

WALL_KEYS = {"wall_bits", "lat_raw_bits", "lat_search_bits", "lat_turn_bits", "game_wall_bits"}


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in WALL_KEYS}


def diff(a, b, path=""):
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        ka, kb = sorted(a.keys()), sorted(b.keys())
        if ka != kb:
            return f"{path}: keys {ka} != {kb}"
        for k in ka:
            d = diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            d = diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("py", nargs="?", default="data/cache/app/product_py.jsonl.gz")
    ap.add_argument("rs", nargs="?", default="data/cache/app/product_rs.jsonl.gz")
    a = ap.parse_args(argv)
    stats = {
        "lines": 0,
        "games": 0,
        "decisions": 0,
        "logs": 0,
        "searched": 0,
        "overrides": 0,
        "det_skipped": 0,
        "capped": 0,
        "branches": 0,
    }
    n = 0
    with gzip.open(a.py, "rt") as fp, gzip.open(a.rs, "rt") as fr:
        while True:
            lp, lr = fp.readline(), fr.readline()
            if not lp and not lr:
                break
            if not lp or not lr:
                print(f"ALERT product_games_compare: line {n + 1}: one side ended early", file=sys.stderr)
                return 2
            try:
                x, y = json.loads(lp), json.loads(lr)
            except json.JSONDecodeError as e:
                print(f"ALERT product_games_compare: line {n + 1}: {e}", file=sys.stderr)
                return 2
            d = diff(strip(x), strip(y))
            if d:
                print(f"ALERT product_games_compare: line {n + 1}: {d}", file=sys.stderr)
                return 2
            n += 1
            stats["lines"] += 1
            kind = x.get("kind")
            if kind == "decision":
                stats["decisions"] += 1
            elif kind == "log":
                stats["logs"] += 1
            elif kind == "game":
                stats["games"] += 1
                stats["searched"] += int(x.get("n_searched", 0))
                stats["overrides"] += int(x.get("n_override", 0))
                stats["det_skipped"] += int(x.get("det_skipped", 0))
                stats["capped"] += int(x.get("n_capped", 0))
                stats["branches"] += int(x.get("n_branches", 0))
    print(
        "OK: {lines} lines identical; games={games} decisions={decisions} logs={logs} searched={searched} "
        "overrides={overrides} det_skipped={det_skipped} capped={capped} branches={branches}".format(**stats)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
