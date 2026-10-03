"""Make verified combo lines stronger (quality pass over plan_lines output).

Usage:
    uv run python scripts/improve_lines.py --deck blue-eyes --src data/demos/planned-s1

Per verified line: prompt the LLM with the card sheet, the exact 5-card hand,
the verified line and its end board, asking for a strictly stronger turn-1
line for the same hand. The candidate is parsed/grounded/verified exactly
like plan_hands; it is kept only if it verifies AND beats the old end board
(bigger, or equal size with more executed steps).

Writes <src>/improved/lines/<deck>.jsonl (kept lines, id "<old id>+imp"),
<src>/improved/verified/<deck>.jsonl (BC records, tag +planned, appended and
deduped by line_id) and <src>/improved/report/<deck>.json (one entry per
attempted old line). Re-runs skip old ids already present in the report.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from ygo_zero.decks import Deck, load_ydk
from ygo_zero.demos import llm as _llm
from ygo_zero.demos.explain import explain_failure
from ygo_zero.demos.extract import _fill_codes, _parse_json
from ygo_zero.demos.names import CardNames
from ygo_zero.demos.plan import (
    SYSTEM,
    card_sheet,
    compact_line_json,
    improve_prompt,
    repair_prompt,
)
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
                logger.warning("improve_lines: skipping malformed line: %s", e)
    return out


def _read_verified(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        if raw.strip():
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return out


def read_report(path: Path) -> list[dict]:
    """Existing per-line improvement report entries ([] when absent/corrupt)."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def attempted_ids(path: Path) -> set[str]:
    """Old ids that already have an improvement attempt recorded."""
    return {str(e.get("old_id")) for e in read_report(path) if e.get("old_id")}


def collect_olds(src_dir: Path, deck_name: str) -> list[tuple[ComboLine, int]]:
    """Verified lines of `deck_name` in `(src_dir)`, in verified-file order.

    Returns `(line, old_executed)` pairs, deduped by line id; verified ids
    with no matching line object are skipped. `old_executed` comes from the
    verified record (`executed_steps`), falling back to `len(line.steps)`.
    """
    lines = _read_lines(src_dir / "lines" / f"{deck_name}.jsonl")
    olds: list[tuple[ComboLine, int]] = []
    seen: set[str] = set()
    for rec in _read_verified(src_dir / "verified" / f"{deck_name}.jsonl"):
        lid = rec.get("line_id")
        if not lid or lid in seen or lid not in lines:
            continue
        seen.add(lid)
        line = lines[lid]
        try:
            old_executed = int(rec.get("executed_steps", len(line.steps)))
        except (TypeError, ValueError):
            old_executed = len(line.steps)
        olds.append((line, old_executed))
    return olds


def _end_size(line: ComboLine) -> int:
    return len(line.end_board_codes or []) or len(line.end_board)


def is_stronger(new_line: ComboLine, new_executed: int, old_line: ComboLine, old_executed: int) -> bool:
    """Strictly-stronger rule: bigger end board, or equal size with more executed steps."""
    new_size, old_size = _end_size(new_line), _end_size(old_line)
    if new_size != old_size:
        return new_size > old_size
    return new_executed > old_executed


def _is_empty_lines_answer(text: str) -> bool:
    """True when the model answers `{"lines": []}` (no stronger line exists).

    `_parse_json` returns [] for both this and unparsable prose, so check the
    raw object explicitly (tolerating fences and surrounding text).
    """
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    try:
        obj = json.loads(t)
        return isinstance(obj, dict) and isinstance(obj.get("lines"), list) and not obj["lines"]
    except json.JSONDecodeError:
        return False


def _ground_candidate(d: dict, hand_names: list[str], deck_name: str, old_id: str, names: CardNames) -> ComboLine:
    """Force hand/going/id and ground names to codes (mirrors plan_hands)."""
    line = ComboLine.from_json(d)
    line.hand = list(hand_names)
    line.going = "first"
    line.id = f"{old_id}+imp"
    line.deck = deck_name
    line.source_file = "planner"
    # card-less narration steps ("pass", "end turn") are not actions: drop them before grounding
    line.steps = [st for st in line.steps if st.card or st.action not in ("other", "pass")]
    for j, st in enumerate(line.steps):
        st.i = j
    _fill_codes(line, names)
    return line


def improve_old_lines(
    engine,
    deck,
    deck_name: str,
    olds: list[tuple[ComboLine, int]],
    names: CardNames,
    complete_many_fn,
    verify_fn,
    repairs: int = 1,
    budget: int = 6000,
    budget_long: int = 15000,
) -> list[dict]:
    """Try one strictly-stronger line per verified old line.

    Round 0 prompts with `improve_prompt`; engine rejections (and unparsable
    answers) go to the next round with `repair_prompt`. A verified-but-weaker
    candidate and an explicit `{"lines": []}` are terminal drops, not repairs.
    Returns one result dict per old line:
    `{old_line, old_executed, attempts:[{line, reason, detail, nodes}], ok_line, ok_verdict}`.
    """
    sheet = card_sheet(engine, deck, names)
    base_prompts = [
        improve_prompt(deck_name, sheet, list(old.hand), compact_line_json(old), list(old.end_board))
        for old, _ in olds
    ]
    results: list[dict] = [
        {"old_line": old, "old_executed": old_exec, "attempts": [], "ok_line": None, "ok_verdict": None,
         "terminal": None}
        for old, old_exec in olds
    ]
    pending = list(range(len(olds)))
    name_of = names.names
    for attempt in range(repairs + 1):
        if not pending:
            break
        prompts: list[str] = []
        for k in pending:
            if attempt == 0:
                prompts.append(base_prompts[k])
            else:
                last = results[k]["attempts"][-1]
                prompts.append(
                    repair_prompt(
                        base_prompts[k],
                        last.get("line_json", ""),
                        f"{last['reason']}: {last['detail']}",
                        last.get("explain", ""),
                    )
                )
        responses = complete_many_fn(prompts)
        # Parse + ground every response; collect the parsed ones for verify.
        to_verify: list[tuple] = []
        verify_pos: list[int] = []
        for k, resp in zip(pending, responses):
            old_line = results[k]["old_line"]
            if isinstance(resp, BaseException):
                results[k]["attempts"].append(
                    {
                        "line": None,
                        "line_json": "",
                        "reason": "unparsable",
                        "detail": f"llm error: {resp}",
                        "nodes": 0,
                        "explain": "",
                    }
                )
                continue
            text = resp if isinstance(resp, str) else str(resp)
            if _is_empty_lines_answer(text):
                results[k]["attempts"].append(
                    {
                        "line": None,
                        "line_json": text[:2000],
                        "reason": "empty",
                        "detail": "model reports no stronger line",
                        "nodes": 0,
                        "explain": "",
                    }
                )
                results[k]["terminal"] = "empty"
                continue
            dicts = _parse_json(text)
            if not dicts:
                results[k]["attempts"].append(
                    {
                        "line": None,
                        "line_json": text[:2000],
                        "reason": "unparsable",
                        "detail": "no ComboLine JSON object found",
                        "nodes": 0,
                        "explain": "",
                    }
                )
                continue
            try:
                line = _ground_candidate(dicts[0], list(old_line.hand), deck_name, old_line.id, names)
            except Exception as e:  # noqa: BLE001 - malformed LLM object -> repair
                results[k]["attempts"].append(
                    {
                        "line": None,
                        "line_json": json.dumps(dicts[0])[:2000],
                        "reason": "unparsable",
                        "detail": f"malformed line: {e}",
                        "nodes": 0,
                        "explain": "",
                    }
                )
                continue
            b = budget_long if len(line.steps) > 8 else budget
            to_verify.append((line, deck, b))
            verify_pos.append(k)
        verdicts = verify_fn(to_verify) if to_verify else []
        lines_by_k = {k: t[0] for k, t in zip(verify_pos, to_verify)}
        for k, v in zip(verify_pos, verdicts):
            # The candidate is the parsed line for this old id (kept in to_verify order).
            line = lines_by_k[k]
            try:
                explain = explain_failure(engine, line, deck, v, name_of)
            except Exception:  # noqa: BLE001 - explain must not kill planning
                explain = ""
            results[k]["attempts"].append(
                {
                    "line": line,
                    "line_json": compact_line_json(line),
                    "reason": v.reason,
                    "detail": v.detail,
                    "nodes": v.nodes,
                    "explain": explain,
                }
            )
            if v.ok:
                results[k]["ok_line"] = line
                results[k]["ok_verdict"] = v
        # Terminal: verified candidates (kept or dropped as weaker later),
        # explicit "no stronger line" answers. The rest stay pending.
        pending = [
            k
            for k in pending
            if results[k]["ok_verdict"] is None and results[k]["terminal"] is None
        ]
    return results


def keep_new(r: dict) -> bool:
    """Stronger-or-drop rule: keep the candidate only if it verified AND beats the old line."""
    return (
        r["ok_line"] is not None
        and r["ok_verdict"] is not None
        and is_stronger(r["ok_line"], r["ok_verdict"].executed_steps, r["old_line"], r["old_executed"])
    )


def append_records_dedupe(path: Path, records: list[dict]) -> int:
    """Append records whose `line_id` is not already in `path`; returns appended count."""
    existing: set[str] = set()
    if path.is_file():
        for raw in path.read_text().splitlines():
            try:
                existing.add(json.loads(raw)["line_id"])
            except (KeyError, json.JSONDecodeError):
                continue
    fresh = [r for r in records if r.get("line_id") not in existing]
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for r in fresh:
                existing.add(r["line_id"])
                f.write(json.dumps(r) + "\n")
    return len(fresh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True, help="pool deck (decks/pool/<NAME>.ydk)")
    ap.add_argument("--src", default="data/demos/planned", help="planner run dir to improve")
    ap.add_argument("--repairs", type=int, default=1)
    ap.add_argument("--model", default="opencode-go/muse-spark-1.3-contributor")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--variant", default=None, help="muse reasoning effort (minimal|low|high|max; default: provider's)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--budget-long", type=int, default=15000)
    ap.add_argument("--limit", type=int, default=None, help="improve at most K lines")
    ap.add_argument(
        "--llm-timeout", type=int, default=900, help="seconds per LLM call (repair prompts are long)"
    )
    ap.add_argument(
        "--remote",
        action="store_true",
        help="verify on the warm vast session box instead of the local fork pool "
        "(start one with scripts/vast_session.py start; LLM calls stay local)",
    )
    args = ap.parse_args()
    if args.remote:
        try:
            require_session()
        except RuntimeError as e:
            ap.error(str(e))

    t0 = time.time()
    src_dir = ROOT / args.src
    out_dir = src_dir / "improved"
    report_path = out_dir / "report" / f"{args.deck}.json"
    done = attempted_ids(report_path)
    olds = [(ln, ex) for ln, ex in collect_olds(src_dir, args.deck) if ln.id not in done]
    if args.limit is not None:
        olds = olds[: max(0, args.limit)]
    if not olds:
        print(f"tried=0 kept=0 mean_end_before=0.0 mean_end_after=0.0 (nothing to improve in {src_dir})")
        return

    engine = Engine()
    names = CardNames()
    deck = load_ydk(ROOT / "decks" / "pool" / f"{args.deck}.ydk")

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
            return remote_verify(jobs, LINE_TIMEOUT)  # the box sizes its pool from its cgroup quota
        _worker_init()  # forked children inherit this Engine
        return map_timeout(
            _verify_job,
            jobs,
            args.jobs,
            LINE_TIMEOUT,
            lambda it, why: failed_verdict(it[0]["id"], it[5], why, LINE_TIMEOUT),
        )

    results = improve_old_lines(
        engine,
        deck,
        args.deck,
        olds,
        names,
        complete_many_fn,
        verify_fn,
        repairs=args.repairs,
        budget=args.budget,
        budget_long=args.budget_long,
    )

    lines_path = out_dir / "lines" / f"{args.deck}.jsonl"
    verified_path = out_dir / "verified" / f"{args.deck}.jsonl"
    for p in (lines_path, verified_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    records: list[dict] = []
    report = read_report(report_path)
    for r in results:
        old = r["old_line"]
        new_id: str | None = None
        new_end: list[str] = []
        if keep_new(r):
            new_id = r["ok_line"].id
            new_end = list(r["ok_line"].end_board)
            r["ok_verdict"].deck_source = "planned"
            records.append(make_record(engine, r["ok_line"], deck, r["ok_verdict"], 0, [0], "+planned"))
            kept += 1
            reason = "kept"
        elif r["ok_line"] is not None:
            last_v = r["ok_verdict"]
            new_end = list(r["ok_line"].end_board)
            reason = f"weaker: verified but end {_end_size(r['ok_line'])} vs {_end_size(old)} ({last_v.reason})"
        else:
            cands = [a for a in r["attempts"] if a["line"] is not None]
            if cands:
                new_end = list(cands[-1]["line"].end_board)
            last = r["attempts"][-1] if r["attempts"] else {"reason": "empty", "detail": "no attempts"}
            reason = f"{last['reason']}: {last['detail']}"[:500]
        report.append(
            {
                "old_id": old.id,
                "new_id": new_id,
                "reason": reason,
                "old_end": list(old.end_board),
                "new_end": new_end,
            }
        )
    with lines_path.open("a") as lf:
        existing_lines = _read_lines(lines_path)
        for rec in records:
            lid = rec["line_id"]
            if lid in existing_lines:
                continue
            line = next(r["ok_line"] for r in results if r["ok_line"] is not None and r["ok_line"].id == lid)
            lf.write(json.dumps(line.to_json()) + "\n")
            existing_lines[lid] = line
    # deck_source="planned" was set on each kept verdict before make_record (as in plan_lines).
    appended = append_records_dedupe(verified_path, records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1))
    wall = time.time() - t0

    before = [len(old.end_board) for old, _ in olds]
    after = []
    for r in results:
        if r["ok_line"] is not None and any(rec["line_id"] == r["ok_line"].id for rec in records):
            after.append(len(r["ok_line"].end_board))
        else:
            after.append(len(r["old_line"].end_board))
    mean_before = sum(before) / len(before)
    mean_after = sum(after) / len(after)
    print(f"tried={len(results)} kept={kept} mean_end_before={mean_before:.1f} mean_end_after={mean_after:.1f}")
    print(f"verified_records={appended} llm_calls={{new={llm_counts['new']} cached={llm_counts['cached']}}} wall={wall:.0f}s")
    print(f"wrote {lines_path}, {verified_path}, report to {report_path}")


if __name__ == "__main__":
    main()
