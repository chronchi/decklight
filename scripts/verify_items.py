"""Verify combo-line / turn-plan items on the (vast session) box.

Reads the items jsonl written by `decklight.infra.remote` (the same tuples `plan_lines._verify_job`
takes, plus {"kind": "plan", ...} dicts for `plan_from_state`), verifies them with `map_timeout`,
and writes one verdict json per line, in input order.

Usage (on the box):
    uv run --no-sync python scripts/verify_items.py --in /tmp/verify_in_X.jsonl \\
        --out /tmp/verify_out_X.jsonl --timeout 900 [--jobs 24]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from decklight.decks import Deck
from decklight.demos.pool import failed_verdict, map_timeout
from decklight.demos.schema import ComboLine
from decklight.demos.verify import Verdict, plan_from_state, verify_line
from decklight.engine.duel import Engine
from decklight.infra.remote import _config_from_json, _deck_from_json, read_items, verdict_to_json

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


def _plan_job(job: dict) -> Verdict:
    """Picklable work item: {"kind": "plan", ...} -> Verdict via plan_from_state (default responder)."""
    engine = _WORKER_ENGINE or Engine()
    config = _config_from_json(job["config"])
    line = ComboLine.from_json(job["line"])
    deck = _deck_from_json(job["deck"])
    return plan_from_state(engine, config, list(job["prefix"]), line, deck, budget=int(job["budget"]))


def _dispatch(job) -> Verdict:
    if isinstance(job, dict) and job.get("kind") == "plan":
        return _plan_job(job)
    return _verify_job(job)


def _fail(job, why: str, timeout: float) -> Verdict:
    if isinstance(job, dict) and job.get("kind") == "plan":
        line_id = job.get("line", {}).get("id", "?") if isinstance(job.get("line"), dict) else "?"
        seed = job.get("config", {}).get("seed", 0) if isinstance(job.get("config"), dict) else 0
        return failed_verdict(line_id, seed, why, timeout)
    return failed_verdict(job[0]["id"], job[5], why, timeout)


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
    ap.add_argument("--in", dest="inp", required=True, help="items jsonl (see decklight.infra.remote)")
    ap.add_argument("--out", dest="outp", required=True, help="verdict jsonl to write, same order as input")
    ap.add_argument("--timeout", type=float, required=True, help="seconds per item before it is killed")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: cgroup cpu quota)")
    args = ap.parse_args()

    items = read_items(args.inp)
    jobs = args.jobs if args.jobs is not None else default_jobs()
    print(f"verify_items: {len(items)} items, jobs={jobs}, timeout={args.timeout}s", flush=True)
    _worker_init()  # forked children inherit this Engine
    verdicts = map_timeout(
        _dispatch,
        items,
        jobs,
        args.timeout,
        lambda it, why: _fail(it, why, args.timeout),
    )
    outp = Path(args.outp)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        for v in verdicts:
            f.write(json.dumps(verdict_to_json(v)) + "\n")
    print(f"verify_items: wrote {len(verdicts)} verdicts to {outp}", flush=True)


if __name__ == "__main__":
    main()
