#!/usr/bin/env python3
"""A3c played gate: the Rust product plays like the Python product, game for game, at scale.

    a3c_gate.py plan  --pairs 150 --units-per-pair 5 --seed 61000 --out runs/<name>/units.txt
    a3c_gate.py check --run runs/<name> [--progress]

Design (user 2026-11-02, "Use the 53 cents available here to make things, do we really need to top up?"): Rust and
Python are bit-identical per game (A3b-1), so an identity check is exact and does not need 1,500 Python games. The gate
plays every unit in Rust and replays the first unit of every pair in the Python oracle (`product_games_py.py`):
  - a unit = one pair, 2 games (game offsets 2k and 2k+1: the product sits in seat 0, then seat 1), deal seeds seed+g;
  - plan order: for every pair (py k=0, rs k=0), then rs k=1..K-1; the box runs units in that order until its
    deadline, so whatever the credit buys is identity coverage first, Rust volume second.
`check` compares every (py, rs) k=0 pair with `product_games_compare.py`'s rules (wall times excluded) and counts the
Rust games, searched decisions, s/search and crashes. Exit 2 on any difference or crashed unit, 3 when no identity
game finished (a gate that compared nothing cannot pass).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "app"))
from product_games_compare import diff, strip  # noqa: E402


def plan(args) -> int:
    pool = sorted(p.stem for p in (ROOT / "decks" / "pool").glob("*.ydk"))
    if len(pool) < 2:
        print("ALERT a3c plan: decks/pool has fewer than 2 decks", file=sys.stderr)
        return 3
    rng = random.Random(args.seed)
    pairs: list[str] = []
    seen: set[tuple[str, str]] = set()
    while len(pairs) < args.pairs:
        a, b = rng.sample(pool, 2)
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append(f"{a}:{b}")
    lines = []
    for p in pairs:
        lines += [f"py {p} 0", f"rs {p} 0"]
    for k in range(1, args.units_per_pair):
        lines += [f"rs {p} {k}" for p in pairs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"{len(pairs)} pairs, {len(lines)} units ({len(pairs)} python) -> {args.out}")
    return 0


def tag(kind: str, pair: str, k: int) -> str:
    return f"{kind}_{pair.replace(':', '__')}_{k}"


def read_rows(path: Path) -> list[dict] | None:
    try:
        with gzip.open(path, "rt") as f:
            return [json.loads(x) for x in f if x.strip()]
    except (OSError, EOFError, json.JSONDecodeError):
        return None


def rc_of(udir: Path, t: str) -> int | None:
    p = udir / f"{t}.rc"
    return int(p.read_text().strip()) if p.exists() else None


def f64(bits) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(bits)))[0]


def check(args) -> int:
    run = Path(args.run)
    udir = run / "u"
    units = [ln.split() for ln in (run / "units.txt").read_text().splitlines() if ln.strip()]
    out = {"units": len(units), "done": {"rs": 0, "py": 0}, "cut": 0, "crashed": [], "skipped": 0,
           "identity_pairs": 0, "identity_games": 0, "identity_decisions": 0, "differences": [],
           "rs_games": 0, "rs_searched": 0, "rs_truncated": 0, "rs_games_by_seat": [0, 0],
           "rs_product_wins_by_seat": [0, 0]}  # fmt: skip
    lat: list[float] = []
    for kind, pair, k in units:
        t = tag(kind, pair, int(k))
        rc = rc_of(udir, t)
        if rc is None:
            continue
        if rc == 124:  # cut by the deadline: a partial file, never compared
            out["cut"] += 1
            continue
        if rc == 200:
            out["skipped"] += 1
            continue
        if rc != 0:
            out["crashed"].append(f"{t} exit {rc}")
            continue
        rows = read_rows(udir / f"{t}.jsonl.gz")
        if not rows:
            out["crashed"].append(f"{t} exit 0 but no rows")
            continue
        out["done"][kind] += 1
        if kind != "rs":
            continue
        for r in rows:
            if r.get("kind") != "game":
                continue
            out["rs_games"] += 1
            out["rs_searched"] += int(r.get("n_searched", 0))
            out["rs_truncated"] += bool(r.get("truncated"))
            lat += [f64(b) for b in r.get("lat_search_bits") or []]
            seat = r.get("s_seat")
            if seat in (0, 1):
                out["rs_games_by_seat"][seat] += 1
                out["rs_product_wins_by_seat"][seat] += r.get("winner") == seat
    for kind, pair, k in units:
        if kind != "py":
            continue
        tp, tr = tag("py", pair, int(k)), tag("rs", pair, int(k))
        if rc_of(udir, tp) != 0 or rc_of(udir, tr) != 0:
            continue
        a, b = read_rows(udir / f"{tp}.jsonl.gz"), read_rows(udir / f"{tr}.jsonl.gz")
        if not a or not b:
            continue
        out["identity_pairs"] += 1
        if len(a) != len(b):
            out["differences"].append(f"{pair}: {len(a)} python lines vs {len(b)} rust lines")
            continue
        bad = None
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            d = diff(strip(x), strip(y))
            if d:
                bad = f"{pair}: line {i}: {d[:3]}"
                break
        if bad:
            out["differences"].append(bad)
        else:
            out["identity_games"] += sum(1 for r in a if r.get("kind") == "game")
            out["identity_decisions"] += sum(1 for r in a if r.get("kind") == "decision")
    if lat:
        lat.sort()
        out["rs_s_per_search"] = {"mean": round(statistics.fmean(lat), 3), "p50": round(lat[len(lat) // 2], 3),
                                  "p90": round(lat[int(len(lat) * 0.9)], 3), "max": round(lat[-1], 3)}  # fmt: skip
    (run / "gate.json").write_text(json.dumps(out, indent=1))
    line = (f"rs units {out['done']['rs']} ({out['rs_games']} games, {out['rs_searched']} searched), "
            f"py units {out['done']['py']}, identity {out['identity_pairs']} pairs / {out['identity_games']} games, "
            f"differences {len(out['differences'])}, crashed {len(out['crashed'])}, cut {out['cut']}")  # fmt: skip
    print(line)
    if args.progress:
        with open(run / "PROGRESS", "a") as f:
            f.write(line + "\n")
    alerts = out["differences"] + out["crashed"]
    if alerts:
        with open(run / "ALERT", "a") as f:
            seen = set((run / "ALERT").read_text().splitlines()) if (run / "ALERT").exists() else set()
            for x in alerts:
                if f"a3c {x}" not in seen:
                    f.write(f"a3c {x}\n")
        return 2
    if not args.progress and out["identity_games"] == 0:
        print("ALERT a3c: no identity game finished", file=sys.stderr)
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--pairs", type=int, default=150)
    p.add_argument("--units-per-pair", type=int, default=5)
    p.add_argument("--seed", type=int, default=61000)
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--run", required=True)
    c.add_argument("--progress", action="store_true", help="append a PROGRESS line; no exit 3 while running")
    args = ap.parse_args(argv)
    return plan(args) if args.cmd == "plan" else check(args)


if __name__ == "__main__":
    sys.exit(main())
