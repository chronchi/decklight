"""Compare the Python and Rust A3a dumps line by line; exit 2 on any difference.

    uv run python scripts/app/search_parts_compare.py data/cache/app/search_parts_py.jsonl.gz \
        data/cache/app/search_parts_rs.jsonl.gz

Prints the counts of games, decisions, forced ties, determinizer runs, proposals and the accept rate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys


def diff(a, b, path=""):
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        ka, kb = list(a.keys()), list(b.keys())
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
    ap.add_argument("py", nargs="?", default="data/cache/app/search_parts_py.jsonl.gz")
    ap.add_argument("rs", nargs="?", default="data/cache/app/search_parts_rs.jsonl.gz")
    a = ap.parse_args(argv)
    n_lines = n_games = n_decisions = n_ties = n_det = n_prop = n_acc = 0
    with gzip.open(a.py, "rt") as fp, gzip.open(a.rs, "rt") as fr:
        while True:
            lp = fp.readline()
            lr = fr.readline()
            if not lp and not lr:
                break
            if not lp or not lr:
                print(
                    f"ALERT search_parts_compare: line {n_lines + 1}: one side ended early", file=sys.stderr
                )
                return 2
            try:
                x = json.loads(lp)
                y = json.loads(lr)
            except json.JSONDecodeError as e:
                print(f"ALERT search_parts_compare: line {n_lines + 1}: {e}", file=sys.stderr)
                return 2
            d = diff(x, y)
            if d:
                print(f"ALERT search_parts_compare: line {n_lines + 1}: {d}", file=sys.stderr)
                return 2
            n_lines += 1
            if x.get("kind") == "end":
                n_games += 1
            elif x.get("kind") == "parts":
                n_decisions += 1
                if x["k"] % 7 == 0:
                    n_ties += 1
            elif x.get("kind") == "det":
                n_det += 1
                n_prop += x["counts"]["proposals"]
                n_acc += x["counts"]["accepted"]
    rate = (n_acc / n_prop) if n_prop else 0.0
    print(
        f"OK: {n_lines} lines identical; games={n_games} decisions={n_decisions} ties={n_ties} "
        f"det_runs={n_det} proposals={n_prop} accepted={n_acc} accept_rate={rate:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
