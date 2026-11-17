"""Compare two gzip JSON-line dumps line by line (A2b-1 parity gate).

    uv run python scripts/app/edopro_compare.py A.jsonl.gz B.jsonl.gz

Exits 0 when every line is equal, 2 with an ALERT naming the first differing line/field otherwise.
Lines are compared as parsed JSON, so key order and whitespace do not matter.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def lines(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        for n, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield n, json.loads(ln)
            except json.JSONDecodeError as e:
                print(f"ALERT edopro_compare: {path}:{n}: unparseable JSON: {e}", file=sys.stderr)
                raise SystemExit(2)


def first_diff(a, b, prefix=""):
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return prefix or "<type>"
    if isinstance(a, dict):
        for k in list(a.keys()) + [k for k in b if k not in a]:
            if k not in a or k not in b:
                return f"{prefix}.{k}" if prefix else k
            d = first_diff(a[k], b[k], f"{prefix}.{k}" if prefix else k)
            if d:
                return d
        return ""
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{prefix}.len"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{prefix}[{i}]")
            if d:
                return d
        return ""
    return "" if a == b else (prefix or "<value>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    a = ap.parse_args(argv)
    ia, ib = lines(a.a), lines(a.b)
    n = 0
    while True:
        try:
            la, va = next(ia)
        except StopIteration:
            la = va = None
        try:
            lb, vb = next(ib)
        except StopIteration:
            lb = vb = None
        if va is None and vb is None:
            break
        n += 1
        if va is None or vb is None:
            print(
                f"ALERT edopro_compare: line count differs at output line {n} "
                f"({a.a.name}:{la} vs {a.b.name}:{lb})",
                file=sys.stderr,
            )
            return 2
        if va != vb:
            field = first_diff(va, vb)
            print(
                f"ALERT edopro_compare: first difference at line {n} "
                f"({va.get('k')} g={va.get('g')} i={va.get('i')} d={va.get('d')}) field {field}",
                file=sys.stderr,
            )
            print(f"  A: {json.dumps(va, sort_keys=True)[:400]}", file=sys.stderr)
            print(f"  B: {json.dumps(vb, sort_keys=True)[:400]}", file=sys.stderr)
            return 2
    print(f"OK: {n} lines identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
