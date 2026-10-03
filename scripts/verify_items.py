"""Verify combo-line items on the (vast session) box.

Reads the items jsonl written by `ygo_zero.infra.remote` (the same tuples `plan_lines._verify_job`
takes), verifies them with `map_timeout`, and writes one verdict json per line, in input order.

Usage (on the box):
    uv run --no-sync python scripts/verify_items.py --in /tmp/verify_in_X.jsonl \\
        --out /tmp/verify_out_X.jsonl --timeout 900 [--jobs 24]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ygo_zero.decks import Deck
from ygo_zero.demos.pool import failed_verdict, map_timeout
from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.verify import Verdict, verify_line
from ygo_zero.engine.duel import Engine
from ygo_zero.infra.remote import read_items, verdict_to_json

_WORKER_ENGINE: Engine | None = None


def _worker_init() -> None:
    """Build one Engine per worker process (Engine is not picklable/thread-safe)."""
    global _WORKER_ENGINE
    _WORKER_ENGINE = Engine()


def _verify_job(job: tuple) -> Verdict:
    """Picklable work item: (line_json, main, extra, side, name, seed, budget) -> Verdict."""
    line_json, main, extra, side, name, seed, budget = job
    engine = _WORKER_ENGINE or Engine()
    line = ComboLine.from_json(line_json)
    deck = Deck(tuple(main), tuple(extra), tuple(side), name)
    return verify_line(engine, line, deck, seed=seed, budget=budget)


def default_jobs() -> int:
    """Worker count from the cgroup cpu quota (cpu_cores_effective), like scripts/vast_expand.sh computes it:
    hosts expose all threads but rent a share."""
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q.strip() == "max":
            return max(1, os.cpu_count() or 1)
        return max(2, int(q) // int(p))
    except (OSError, ValueError, ZeroDivisionError):
        return max(1, min(4, os.cpu_count() or 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="items jsonl (see ygo_zero.infra.remote)")
    ap.add_argument("--out", dest="outp", required=True, help="verdict jsonl to write, same order as input")
    ap.add_argument("--timeout", type=float, required=True, help="seconds per item before it is killed")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: cgroup cpu quota)")
    args = ap.parse_args()

    items = read_items(args.inp)
    jobs = args.jobs if args.jobs is not None else default_jobs()
    print(f"verify_items: {len(items)} items, jobs={jobs}, timeout={args.timeout}s", flush=True)
    _worker_init()  # forked children inherit this Engine
    verdicts = map_timeout(
        _verify_job,
        items,
        jobs,
        args.timeout,
        lambda it, why: failed_verdict(it[0]["id"], it[5], why, args.timeout),
    )
    outp = Path(args.outp)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        for v in verdicts:
            f.write(json.dumps(verdict_to_json(v)) + "\n")
    print(f"verify_items: wrote {len(verdicts)} verdicts to {outp}", flush=True)


if __name__ == "__main__":
    main()
