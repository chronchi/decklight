"""ADR 0002 stage 4: expand verified lines over different opening hands.

For each verified record, deal K hand variants: the line's own hand + random other cards from the same deck up to
`--hand-size`, then re-verify the line from scratch (the search re-derives every choice). Only variants that
verify are kept, tagged `variant` in their line_id. Different hands give genuinely different observations; on
turn 1 other seeds alone would not (the own deck is an unordered multiset in the obs).

    uv run python scripts/expand_lines.py --verified data/demos/verified --lines-dir data/demos/lines \
        --variants 8 --out data/demos/expanded
"""

import argparse
import dataclasses
import glob
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from ygo_zero.decks import Deck
from ygo_zero.demos.dataset import load_records
from ygo_zero.demos.pool import failed_verdict, map_timeout
from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.verify import Verdict, verify_line
from ygo_zero.engine.duel import Engine
from ygo_zero.infra.remote import remote_verify


def default_jobs() -> int:
    """Default worker count: up to 4 procs (verify is CPU-bound; more thrash)."""
    return max(1, min(4, os.cpu_count() or 1))


_WORKER_ENGINE: Engine | None = None


def _worker_init() -> None:
    """Build one Engine per worker process (Engine is not picklable/thread-safe)."""
    global _WORKER_ENGINE
    _WORKER_ENGINE = Engine()


def _expand_job(job: tuple) -> Verdict:
    """Picklable work item: (var_json, main, extra, seed, budget) -> Verdict."""
    var_json, main, extra, seed, budget = job
    engine = _WORKER_ENGINE or Engine()
    var = ComboLine.from_json(var_json)
    deck = Deck(tuple(main), tuple(extra))
    return verify_line(engine, var, deck, seed=seed, budget=budget)


VARIANT_TIMEOUT = 600.0  # seconds per hand variant; see demos/pool.py


def base_id_of(line_id: str) -> str:
    """The part of an expansion's line_id before `#` (`<base id>#<suffix>`)."""
    return line_id.split("#")[0]


def collect_done_ids(patterns: list[str]) -> set[str]:
    """Base ids that already have expansions in any file matching the globs."""
    done: set[str] = set()
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            p = Path(f)
            if not p.is_file():
                continue
            for raw in p.read_text().splitlines():
                if not raw.strip():
                    continue
                try:
                    done.add(base_id_of(json.loads(raw)["line_id"]))
                except (KeyError, json.JSONDecodeError):
                    continue
    return done


def map_ordered(fn, items: list, jobs: int, initializer=None):
    """Run fn over items in input order, each in a forked child with a wall-clock limit (a killed job fails)."""
    if not items:
        return []
    if initializer is not None:
        initializer()  # children inherit the parent's Engine
    return map_timeout(
        fn,
        items,
        jobs,
        VARIANT_TIMEOUT,
        lambda it, why: failed_verdict(it[0]["id"], it[3], why, VARIANT_TIMEOUT),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", type=Path, default=Path("data/demos/verified"))
    ap.add_argument("--lines-dir", type=Path, default=Path("data/demos/lines"))
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--hand-size", type=int, default=5)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/demos/expanded"))
    ap.add_argument(
        "--skip-done",
        nargs="*",
        default=[],
        metavar="GLOB",
        help="skip input records whose line_id already has expansions in files matching these globs",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="worker processes over hand variants (default: %(default)s)",
    )
    ap.add_argument(
        "--remote",
        action="store_true",
        help="verify variants on the warm session box via remote_verify (one call for the whole batch)",
    )
    ap.add_argument("--session", default="lines", help="vast session name for --remote")
    args = ap.parse_args()

    lines = {}
    for f in args.lines_dir.glob("*.jsonl"):
        for raw in f.read_text().splitlines():
            if raw.strip():
                d = json.loads(raw)
                lines[d["id"]] = d
    args.out.mkdir(parents=True, exist_ok=True)
    # --skip-done: incremental (append; the skip set is read BEFORE writing). Without it: rewrite from scratch.
    done = collect_done_ids(args.skip_done) if args.skip_done else set()
    if not args.skip_done:
        for old in args.out.glob("*.jsonl"):
            old.unlink()
    rng = random.Random(args.seed)
    kept = Counter()
    t0 = time.time()
    skipped = 0
    # Pass 1: deal every record's hand variants (rng order unchanged), so pass 2 can verify all of them in one pool
    # call; one record's variants at a time left most cores idle on big boxes (8 busy of 23).
    plans = []  # (rec, me, variants, full_main, my_extra)
    for rec in load_records(args.verified):
        # improved lines keep "+imp" in their own id; other records carry a "+tag" suffix not in the line id
        base_id = rec["line_id"] if rec["line_id"] in lines else rec["line_id"].split("+")[0]
        if base_id_of(rec["line_id"]) in done or base_id in done:
            skipped += 1
            continue
        if base_id not in lines:
            print(f"{rec['line_id']}: source line not found, skipped")
            continue
        line = ComboLine.from_json(lines[base_id])
        c = rec["config"]
        hands = [p for p in c["setup"] if p[2] == 2]  # LOCATION_HAND placements: whose they are tells my seat
        me = hands[0][0] if hands else 0  # going second → team 1
        hand = [p[1] for p in hands]
        my_main, my_extra = c[f"deck{me}_main"], c[f"deck{me}_extra"]
        full_main = list(my_main) + hand  # the record's deck excludes the placed hand
        variants: list = []
        for v in range(args.variants):
            pool = list(my_main)
            rng.shuffle(pool)
            extra = pool[: max(0, args.hand_size - len(hand))]
            var = dataclasses.replace(
                line, id=f"{base_id}#h{v}", hand=line.hand + [str(x) for x in extra], hand_codes=hand + extra
            )
            variants.append((var, extra))
        plans.append((rec, me, variants, full_main, my_extra))
    items = [
        (var.to_json(), tuple(full_main), tuple(my_extra), rec["config"]["seed"], args.budget)
        for rec, _, variants, full_main, my_extra in plans
        for var, _ in variants
    ]
    print(f"{len(plans)} records, {len(items)} hand variants to verify", flush=True)
    if args.skip_done:
        print(f"skipped {skipped} records with existing expansions", flush=True)
    if args.remote:
        # Same verify-item 7-tuples plan_lines sends (side/name default to ()/""): verify_items._verify_job
        # rebuilds Deck(main, extra, side, name) and runs the same verify_line, so no new item kind is needed.
        remote_items = [
            (var.to_json(), tuple(full_main), tuple(my_extra), (), rec.get("deck", ""),
             rec["config"]["seed"], args.budget)
            for rec, _, variants, full_main, my_extra in plans
            for var, _ in variants
        ]
        print(f"verifying {len(remote_items)} variants on session '{args.session}'", flush=True)
        flat = remote_verify(remote_items, VARIANT_TIMEOUT, session=args.session)
    else:
        flat = map_ordered(_expand_job, items, args.jobs, initializer=_worker_init)
    # Pass 2: write the verified variants, per record, in input order.
    pos = 0
    for rec, me, variants, _, _ in plans:
        verdicts = flat[pos : pos + len(variants)]
        pos += len(variants)
        out_records = []
        for (var, extra), verdict in zip(variants, verdicts):
            kept[verdict.reason] += 1
            if not verdict.ok:
                continue
            cfg = verdict.config
            out_records.append({
                **{k: rec[k] for k in ("source_url", "deck", "end_board", "end_board_codes") if k in rec},
                "line_id": var.id,
                "variant_of": rec["line_id"],
                "me": me,
                "config": {
                    "deck0_main": list(cfg.deck0.main), "deck0_extra": list(cfg.deck0.extra),
                    "deck1_main": list(cfg.deck1.main), "deck1_extra": list(cfg.deck1.extra),
                    "seed": cfg.seed, "hand": cfg.hand, "draw": cfg.draw,
                    "setup": [[p.team, p.code, p.location, p.sequence] for p in cfg.setup],
                },
                "auto_single": True,
                "choices": verdict.choices,
                "labels": verdict.labels,
                "skipped_steps": list(verdict.skipped_steps or []),
                "executed_steps": verdict.executed_steps,
                "inferred_frac": verdict.inferred_frac,
                "extra_hand": extra,
            })  # fmt: skip
        with open(args.out / f"{rec['deck']}.jsonl", "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in out_records)
        print(f"{rec['line_id']}: {len(out_records)}/{args.variants} hand variants verify", flush=True)
    print(f"verdicts: {dict(kept)}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
