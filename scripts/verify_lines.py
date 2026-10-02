"""Verify extracted combo lines in the engine (ADR 0002 stage 3).

Usage:
    uv run python scripts/verify_lines.py --lines data/demos/lines/blue-eyes.jsonl \\
        --ydk tests/fixtures/decks/BlueEyes.ydk [--seeds 0 1 2] --out data/demos/verified/blue-eyes.jsonl

    --ydk may be omitted: it then defaults to decks/pool/<line.deck>.ydk when
    that file exists (all four demo decks have one).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ygo_zero.decks import Deck, load_ydk
from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.verify import Verdict, going_second, make_config, verify_line
from ygo_zero.engine import constants as C
from ygo_zero.engine.duel import Engine

ROOT = Path(__file__).resolve().parents[1]

_EXTRA = C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK


def default_jobs() -> int:
    """Default worker count: up to 4 procs (verify is CPU-bound; more thrash)."""
    return max(1, min(4, os.cpu_count() or 1))


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


def map_ordered(fn, items: list, jobs: int, initializer=None):
    """Run fn over items, returning results in input order (identical to serial)."""
    if jobs <= 1 or not items:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=jobs, initializer=initializer) as ex:
        return list(ex.map(fn, items))


def default_pool_ydk(deck: str) -> Path | None:
    """Tracked current-TCG list for `deck`, if one exists (`decks/pool/<deck>.ydk`)."""
    cand = ROOT / "decks" / "pool" / f"{deck}.ydk"
    return cand if cand.is_file() else None


def augmented(engine: Engine, deck: Deck, line: ComboLine) -> Deck:
    """Fixture deck + the line's hand/required codes (extra-deck types go to the extra deck)."""
    main, extra = list(deck.main), list(deck.extra)
    for c in line.hand_codes + line.requires_codes:
        if c is None:
            continue
        cd = engine.cards.get(c)
        if cd is not None and cd.type & _EXTRA:
            extra.append(c)
        else:
            main.append(c)
    return Deck(tuple(main), tuple(extra), deck.side, deck.name + "+line")


def stats(v: Verdict) -> str:
    mine = [x for x in v.labels if x != "auto"]
    inf = sum(x == "inferred" for x in mine)
    pct = 100.0 * inf / len(mine) if mine else 0.0
    return f"{pct:.0f}% inferred ({inf}/{len(mine)})"


def verify_all(
    engine: Engine,
    deck: Deck,
    lines: list[ComboLine],
    seeds: list[int],
    budget: int,
    tag: str,
    deck_source: str = "",
    budget_long: int | None = None,
    jobs: int = 1,
) -> tuple[list[dict], dict[str, Verdict]]:
    """Verify every line under every seed; print per-line results; return ok records + reps.

    Lines with more than 8 steps use budget_long when given (long lines need more search nodes).
    With jobs > 1, (line, seed) pairs verify in worker processes (one Engine each);
    results are regrouped in serial order so output is identical to jobs=1.
    """
    main_t, extra_t, side_t = tuple(deck.main), tuple(deck.extra), tuple(deck.side)
    items: list[tuple] = []
    for line in lines:
        b = budget_long if budget_long and len(line.steps) > 8 else budget
        for s in seeds:
            items.append((line.to_json(), main_t, extra_t, side_t, deck.name, s, b))
    if jobs <= 1:
        # serial path reuses the caller's engine directly (same order as the job list)
        flat = []
        for line in lines:
            b = budget_long if budget_long and len(line.steps) > 8 else budget
            for s in seeds:
                flat.append(verify_line(engine, line, deck, seed=s, budget=b))
    else:
        flat = map_ordered(_verify_job, items, jobs, initializer=_worker_init)
    records, reps = [], {}
    pos = 0
    for line in lines:
        verdicts = [(seeds[i], flat[pos + i]) for i in range(len(seeds))]
        pos += len(seeds)
        ok = [(s, v) for s, v in verdicts if v.ok]
        rep = ok[0][1] if ok else verdicts[0][1]
        rep.deck_source = deck_source or tag.lstrip("+") or "fixture"
        reps[line.id] = rep
        print(
            f"{line.id}{tag}: {rep.reason} steps={len(line.steps)} executed={rep.executed_steps} "
            f"nodes={rep.nodes} {stats(rep)} inferred_frac={rep.inferred_frac:.2f} "
            f"seeds_ok={[s for s, _ in ok]} skipped={rep.skipped_steps} deck={rep.deck_source} "
            f"hand_inferred={rep.hand_inferred}"
        )
        print(f"    {rep.detail}")
        if ok:
            s0, v0 = ok[0]
            cfg = v0.config or make_config(line, deck, s0, engine)
            records.append(
                {
                    "line_id": line.id,
                    "me": 1 if going_second(line) else 0,
                    "source_url": line.source_url,
                    "deck": line.deck + tag,
                    "deck_source": v0.deck_source,
                    "config": {
                        "deck0_main": list(cfg.deck0.main),
                        "deck0_extra": list(cfg.deck0.extra),
                        "deck1_main": list(cfg.deck1.main),
                        "deck1_extra": list(cfg.deck1.extra),
                        "seed": s0,
                        "hand": cfg.hand,
                        "draw": cfg.draw,
                        "setup": [[p.team, p.code, p.location, p.sequence] for p in cfg.setup],
                    },
                    "auto_single": True,
                    "choices": v0.choices,
                    "labels": v0.labels,
                    "skipped_steps": list(v0.skipped_steps or []),
                    "executed_steps": v0.executed_steps,
                    "inferred_frac": v0.inferred_frac,
                    "hand_inferred": v0.hand_inferred,
                    "end_board": line.end_board,
                    "end_board_codes": line.end_board_codes,
                    "seeds_ok": [s for s, _ in ok],
                }
            )
    return records, reps


def deck_from_guide_json(engine: Engine, data: dict, name: str) -> Deck:
    """Build a Deck from a guide-decklist JSON (re-split by card type)."""
    main, extra = [], []
    for c in list(data.get("main", [])) + list(data.get("extra", [])):
        cd = engine.cards.get(int(c))
        if cd is not None and cd.type & _EXTRA:
            extra.append(int(c))
        else:
            main.append(int(c))
    return Deck(tuple(main), tuple(extra), (), name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument(
        "--ydk",
        required=False,
        default=None,
        help="fixture deck (default: decks/pool/<line.deck>.ydk when it exists)",
    )
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument(
        "--budget-long",
        type=int,
        default=None,
        help="node budget for lines with more than 8 steps (default: same as --budget)",
    )
    ap.add_argument(
        "--deck-from-guides",
        default=None,
        help="per-guide decklists dir (<dir>/<deck>/<source>.json); each line is verified "
        "against its own source's decklist when present, else the --ydk fixture",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="worker processes over (line, seed) pairs (default: %(default)s)",
    )
    args = ap.parse_args()

    engine = Engine()
    lines = [
        ComboLine.from_json(json.loads(s)) for s in Path(args.lines).read_text().splitlines() if s.strip()
    ]
    if not lines:
        raise SystemExit(f"no lines in {args.lines}")
    ydk_arg = args.ydk
    if ydk_arg is None:
        pool = default_pool_ydk(lines[0].deck)
        if pool is None:
            raise SystemExit(
                f"no --ydk given and no decks/pool/{lines[0].deck}.ydk exists; pass --ydk explicitly"
            )
        ydk_arg = str(pool)
    deck = load_ydk(ydk_arg)

    # group lines by deck: guide decklist when available, else the fixture
    groups: dict[str, tuple[Deck, list]] = {}

    def group_for(line: ComboLine) -> tuple[str, Deck]:
        if args.deck_from_guides:
            stem = Path(line.source_file).stem
            jp = Path(args.deck_from_guides) / line.deck / f"{stem}.json"
            if jp.is_file():
                gd = deck_from_guide_json(engine, json.loads(jp.read_text()), f"{line.deck}+guide:{stem}")
                return f"guide:{stem}", gd
        return "fixture", deck

    for line in lines:
        key, gd = group_for(line)
        if key not in groups:
            groups[key] = (gd, [])
        groups[key][1].append(line)

    print(f"== {args.lines} vs {ydk_arg} (seeds={args.seeds}) deck_groups={sorted(groups)} ==")
    records: list[dict] = []
    reps: dict[str, Verdict] = {}
    base_of: dict[str, str] = {}  # line_id -> group key (for the augmented fallback source)
    for key in sorted(groups):
        gd, glines = groups[key]
        tag = "" if key == "fixture" else "+guide"
        print(f"-- deck group {key} ({len(glines)} lines) --")
        recs, greps = verify_all(
            engine, gd, glines, args.seeds, args.budget, tag, key, args.budget_long, args.jobs
        )
        records += recs
        reps.update(greps)
        for line in glines:
            base_of[line.id] = key

    verified = {r["line_id"] for r in records}
    aug_lines = [line for line in lines if line.id not in verified and reps[line.id].reason == "missing_card"]
    aug_records: list[dict] = []
    if aug_lines:
        print(f"== augmented deck ({len(aug_lines)} lines with missing_card) ==")
        for line in aug_lines:
            base_key = base_of[line.id]
            base_deck = groups[base_key][0]
            adeck = augmented(engine, base_deck, line)
            tag = "+augmented" if base_key == "fixture" else "+guide+augmented"
            recs, areps = verify_all(
                engine,
                adeck,
                [line],
                args.seeds,
                args.budget,
                tag,
                base_key + "+augmented",
                args.budget_long,
                args.jobs,
            )
            aug_records += recs
            reps[line.id + "+augmented"] = areps[line.id]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("".join(json.dumps(r) + "\n" for r in records + aug_records))
    print(f"wrote {len(records) + len(aug_records)} verified records to {outp}")

    totals = Counter()
    ok_aug = {r["line_id"] for r in aug_records}
    for line in lines:
        if line.id in verified:
            totals[f"ok[{base_of[line.id]}]"] += 1
        elif line.id in ok_aug:
            totals[f"ok[{base_of[line.id]}+augmented]"] += 1
        else:
            totals[reps[line.id].reason] += 1
    print("totals:", dict(totals))


if __name__ == "__main__":
    main()
