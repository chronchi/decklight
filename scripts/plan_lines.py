"""Plan turn-1 combo lines for sampled hands with an LLM; verify with the engine.

Usage:
    uv run python scripts/plan_lines.py --deck blue-eyes --hands 20 --seed 0 \\
        --out-dir data/demos/planned

Writes <out-dir>/lines/<deck>.jsonl (every attempt's line),
<out-dir>/verified/<deck>.jsonl (BC records, tag +planned) and
<out-dir>/report/<deck>.json (per-hand attempts and reasons).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

from ygo_zero.decks import Deck, load_ydk
from ygo_zero.demos import llm as _llm
from ygo_zero.demos.names import CardNames
from ygo_zero.demos.plan import SYSTEM, plan_hands, sample_hands
from ygo_zero.demos.pool import failed_verdict, map_timeout
from ygo_zero.demos.records import make_record
from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.verify import Verdict, verify_line
from ygo_zero.engine.duel import Engine
from ygo_zero.infra.remote import remote_verify, require_session

ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)

_WORKER_ENGINE: Engine | None = None
LINE_TIMEOUT = 900.0  # seconds per verify job; see demos/pool.py


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


def _read_lines(path: Path) -> dict[str, ComboLine]:
    out: dict[str, ComboLine] = {}
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        if raw.strip():
            try:
                d = json.loads(raw)
                out[d["id"]] = ComboLine.from_json(d)
            except Exception as e:  # noqa: BLE001 - skip malformed lines
                logger.warning("plan_lines: skipping malformed line: %s", e)
    return out


def _verified_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids = set()
    for raw in path.read_text().splitlines():
        try:
            ids.add(json.loads(raw)["line_id"])
        except (KeyError, json.JSONDecodeError):
            continue
    return ids


def load_examples(deck_name: str, k: int) -> list[ComboLine]:
    """Engine-verified lines of the same deck from every source (guides, Claude, every planner run), biggest end
    board first, else verified lines of other decks. New decks thus bootstrap from their own verified lines."""
    if k <= 0:
        return []
    demos = ROOT / "data" / "demos"
    dirs = [demos, demos / "claude", *sorted(demos.glob("planned*"))]
    same: dict[str, ComboLine] = {}
    for d in dirs:
        for sub in ("", "improved"):
            base = d / sub if sub else d
            lines = _read_lines(base / "lines" / f"{deck_name}.jsonl")
            for i in _verified_ids(base / "verified" / f"{deck_name}.jsonl"):
                if i in lines:
                    same[i] = lines[i]
    if same:
        return sorted(same.values(), key=lambda ln: (-len(ln.end_board), -len(ln.steps), ln.id))[:k]
    # Fallback: the strongest verified guide/Claude lines of other decks, one per deck.
    best: dict[str, ComboLine] = {}
    for d in (demos, demos / "claude"):
        for f in sorted((d / "lines").glob("*.jsonl")):
            ids = _verified_ids(d / "verified" / f.name)
            for i, ln in _read_lines(f).items():
                cur = best.get(f.stem)
                if i in ids and (cur is None or len(ln.end_board) > len(cur.end_board)):
                    best[f.stem] = ln
    return sorted(best.values(), key=lambda ln: (-len(ln.end_board), ln.id))[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True, help="deck name (<deck-dir>/<NAME>.ydk)")
    ap.add_argument("--deck-dir", default="decks/pool",
                    help="decks/newcards etc.: eval-only decks; then write --out-dir outside data/demos (BC globs it)")
    ap.add_argument("--hands", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--model", default="opencode/muse-spark-1.3-contributor-free")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--variant", default=None, help="muse reasoning effort (minimal|low|high|max; default: provider's)"
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--budget-long", type=int, default=15000)
    ap.add_argument("--examples", type=int, default=2)
    ap.add_argument(
        "--candidates", type=int, default=1,
        help="best-of-N: ask for N different lines per hand in one call, keep the strongest verified",
    )
    ap.add_argument(
        "--session", default="lines",
        help="vast session name for --remote verification (driver spreads work over boxes)",
    )
    ap.add_argument(
        "--llm-timeout", type=int, default=900, help="seconds per LLM call (repair prompts are long)"
    )
    ap.add_argument("--out-dir", default="data/demos/planned")
    ap.add_argument(
        "--remote",
        action="store_true",
        help="verify on the warm vast session box instead of the local fork pool "
        "(start one with scripts/vast_session.py start; LLM calls stay local)",
    )
    args = ap.parse_args()
    if args.remote:
        try:
            require_session(args.session)
        except RuntimeError as e:
            ap.error(str(e))

    t0 = time.time()
    engine = Engine()
    names = CardNames()
    if args.deck_dir != "decks/pool" and args.out_dir.startswith("data/demos"):
        ap.error("eval-only decks must not write lines under data/demos (build_bc_shards trains on them)")
    deck = load_ydk(ROOT / args.deck_dir / f"{args.deck}.ydk")
    hands = sample_hands(deck, args.hands, args.seed)
    examples = load_examples(args.deck, args.examples)
    print(f"examples: {len(examples)} ({[e.id for e in examples]})")

    llm_counts = {"new": 0, "cached": 0}

    def complete_many_fn(prompts: list[str]) -> list:
        for p in prompts:
            key = _llm._cache_key(args.model, SYSTEM, p, args.variant or _llm.DEFAULT_VARIANT)
            if (_llm.CACHE_DIR / f"{key}.json").exists():
                llm_counts["cached"] += 1
            else:
                llm_counts["new"] += 1
        return _llm.complete_many(
            prompts,
            model=args.model,
            system=SYSTEM,
            max_workers=args.workers,
            return_exceptions=True,
            timeout=args.llm_timeout,
            variant=args.variant,
        )

    main_t, extra_t, side_t = tuple(deck.main), tuple(deck.extra), tuple(deck.side)

    def verify_fn(items: list[tuple]) -> list[Verdict]:
        jobs = [(ln.to_json(), main_t, extra_t, side_t, deck.name, 0, b) for ln, _, b in items]
        if not jobs:
            return []
        if args.remote:
            return remote_verify(jobs, LINE_TIMEOUT, session=args.session)  # box sizes pool from cgroup quota
        _worker_init()  # forked children inherit this Engine
        return map_timeout(
            _verify_job,
            jobs,
            args.jobs,
            LINE_TIMEOUT,
            lambda it, why: failed_verdict(it[0]["id"], it[5], why, LINE_TIMEOUT),
        )

    results = plan_hands(
        engine,
        deck,
        args.deck,
        hands,
        examples,
        names,
        complete_many_fn,
        verify_fn,
        repairs=args.repairs,
        budget=args.budget,
        budget_long=args.budget_long,
        id_tag=f"-s{args.seed}" if args.seed else "",
        candidates=args.candidates,
    )

    out_dir = ROOT / args.out_dir
    lines_path = out_dir / "lines" / f"{args.deck}.jsonl"
    verified_path = out_dir / "verified" / f"{args.deck}.jsonl"
    report_path = out_dir / "report" / f"{args.deck}.json"
    for p in (lines_path, verified_path, report_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    report: list[dict] = []
    by_attempt = Counter()
    reasons = Counter()
    exec_steps: list[int] = []
    bc_decisions: list[int] = []
    with lines_path.open("w") as lf:
        for k, r in enumerate(results):
            ok_attempt: int | None = None
            rep_attempts = []
            for a, att in enumerate(r["attempts"]):
                reasons[att["reason"]] += 1
                entry = {
                    "attempt": a,
                    "line_id": att["line"].id if att["line"] is not None else None,
                    "reason": att["reason"],
                    "detail": att["detail"][:500],
                    "nodes": att["nodes"],
                }
                rep_attempts.append(entry)
                if att["line"] is not None:
                    lf.write(json.dumps(att["line"].to_json()) + "\n")
            if r["ok_verdict"] is not None:
                best_id = r["ok_line"].id if r["ok_line"] is not None else None
                ok_attempt = next(
                    (a for a, att in enumerate(r["attempts"]) if att["reason"] == "ok"
                     and att["line"] is not None and att["line"].id == best_id),
                    next(a for a, att in enumerate(r["attempts"]) if att["reason"] == "ok"),
                )
                by_attempt[ok_attempt] += 1
                v = r["ok_verdict"]
                v.deck_source = "planned"
                records.append(make_record(engine, r["ok_line"], deck, v, 0, [0], "+planned"))
                exec_steps.append(v.executed_steps)
                bc_decisions.append(sum(1 for x in v.labels if x != "auto"))
                # Best-of-N extras: other verified candidates with a distinct end
                # board (ids already carry the -c<k> tag from plan_hands).
                for x in r.get("extras", []):
                    xv = x["verdict"]
                    xv.deck_source = "planned"
                    records.append(make_record(engine, x["line"], deck, xv, 0, [0], "+planned"))
                    exec_steps.append(xv.executed_steps)
                    bc_decisions.append(sum(1 for lab in xv.labels if lab != "auto"))
            report.append(
                {
                    "hand": list(r["hand"]),
                    "hand_names": [names.names.get(c, str(c)) for c in r["hand"]],
                    "attempts": rep_attempts,
                    "ok_attempt": ok_attempt,
                }
            )
    verified_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    report_path.write_text(json.dumps(report, indent=1))
    wall = time.time() - t0

    mean_exec = sum(exec_steps) / len(exec_steps) if exec_steps else 0.0
    mean_bc = sum(bc_decisions) / len(bc_decisions) if bc_decisions else 0.0
    print(
        f"hands={len(results)} verified={len(records)} by_attempt={dict(sorted(by_attempt.items()))} "
        f"mean_executed={mean_exec:.1f} mean_bc_decisions={mean_bc:.1f}"
    )
    print(f"reasons={dict(reasons.most_common())}")
    print(f"llm_calls={{new={llm_counts['new']} cached={llm_counts['cached']}}} wall={wall:.0f}s")
    print(f"wrote {lines_path}, {len(records)} records to {verified_path}, report to {report_path}")


if __name__ == "__main__":
    main()
