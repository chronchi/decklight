"""LLM combo planner — "LLM plans, engine verifies" (Phase 3 data source).

Sample opening hands from a deck; an LLM writes a turn-1 combo line per hand
in the ComboLine schema; the engine verifier checks it; failures come back
with a repair prompt (≤ N repairs). Verified lines become BC records.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import Counter
from collections.abc import Callable

from ygo_zero.demos.explain import explain_failure
from ygo_zero.demos.extract import _fill_codes, _parse_json
from ygo_zero.demos.names import CardNames
from ygo_zero.demos.schema import ACTIONS, ComboLine
from ygo_zero.demos.strategy import notes_block
from ygo_zero.engine import constants as C
from ygo_zero.engine.resources import default_cdbs

SYSTEM = (
    "You are a Yu-Gi-Oh! combo analyst. Write the strongest turn-1 combo line "
    "the given hand allows as strict JSON. Use exact card names as written. "
    "Never invent card IDs — output names only. Reply with JSON only, no prose."
)

# Same schema paragraph + results-attribution rules as extract._prompt
# (copied verbatim; do NOT edit extract._prompt — its exact string is the LLM
# cache key). The planner asks for one going-first line instead of all routes
# in a section.
SCHEMA_TEXT = """Each ComboLine has: title, hand (starting-hand card names), requires (other cards the line needs in deck/extra), steps (ordered; each step: i (0-based), action (one of: {})), card, effect (free-text hint), materials, targets, results (name lists), text (original sentence)), end_board (card names on field), end_board_text, claimed_interruptions (or null), going (first|second|unknown).
Output card NAMES only; omit all *_codes fields (filled automatically). A step's card may be "" with action "other" for game actions without a card (e.g. "send equipped card to GY").

CRITICAL — results attribution: a step's `results` = ONLY the cards that THIS step itself puts into
the hand, field, GY or banished pile when it resolves. Cards that arrive in LATER steps (End Phase
triggers, follow-up searches/summons) belong to those later steps — NEVER copy them into the
current step's results. Likewise `materials` = only cards consumed AS COST/MATERIAL by this step.""".format(
    ", ".join(ACTIONS)
)


def sample_hands(deck, n: int, seed: int, size: int = 5) -> list[list[int]]:
    """Reproducible random opening hands: draws from `deck.main` without replacement within a hand."""
    rng = random.Random(seed)
    main = list(deck.main)
    return [rng.sample(main, size) for _ in range(n)]


def _desc_map() -> dict[int, str]:
    out: dict[int, str] = {}
    for path in default_cdbs():
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for code, desc in con.execute("SELECT id, desc FROM texts").fetchall():
                out.setdefault(code, desc or "")
        finally:
            con.close()
    return out


def _kind_name(t: int) -> str:
    if t & C.TYPE_LINK:
        return "Link Monster"
    if t & C.TYPE_XYZ:
        return "Xyz Monster"
    if t & C.TYPE_SYNCHRO:
        return "Synchro Monster"
    if t & C.TYPE_FUSION:
        return "Fusion Monster"
    if t & C.TYPE_RITUAL:
        return "Ritual Monster"
    if t & C.TYPE_MONSTER:
        return "Monster"
    if t & C.TYPE_SPELL:
        return "Spell"
    if t & C.TYPE_TRAP:
        return "Trap"
    return f"type{t}"


def _attr_name(a: int) -> str:
    return {
        C.ATTRIBUTE_EARTH: "EARTH",
        C.ATTRIBUTE_WATER: "WATER",
        C.ATTRIBUTE_FIRE: "FIRE",
        C.ATTRIBUTE_WIND: "WIND",
        C.ATTRIBUTE_LIGHT: "LIGHT",
        C.ATTRIBUTE_DARK: "DARK",
        C.ATTRIBUTE_DIVINE: "DIVINE",
    }.get(a, str(a))


def _race_name(r: int) -> str:
    for key in dir(C):
        if key.startswith("RACE_") and getattr(C, key) == r and r not in (0,):
            return key[len("RACE_") :].title()
    return str(r)


def _stat_line(code: int, cards, desc: str) -> str:
    cd = cards.get(code)
    if cd is None:
        return f"ATK?/DEF? | {desc}"
    if cd.type & C.TYPE_LINK:
        lv = f"Link-{cd.link_marker.bit_count()}"
    elif cd.type & C.TYPE_XYZ:
        lv = f"Rank {cd.level}"
    elif cd.type & C.TYPE_MONSTER:
        lv = f"Lv {cd.level}"
    else:
        lv = "-"
    return f"{_kind_name(cd.type)}/{_attr_name(cd.attribute)}/{_race_name(cd.race)}, {lv}, ATK{cd.attack}/DEF{cd.defense} | {desc}"


def card_sheet(engine, deck, names: CardNames) -> str:
    """One line per distinct card in main+extra: name, copies, stats, effect text."""
    descs = _desc_map()
    counts = Counter(list(deck.main) + list(deck.extra))
    lines = []
    for code in sorted(counts, key=lambda c: names.names.get(c, str(c))):
        name = names.names.get(code, str(code))
        raw = descs.get(code, "") or ""
        desc = re.sub(r"\s+", " ", raw).strip()[:400]
        lines.append(f"{name} x{counts[code]} | {_stat_line(code, engine.cards, desc)}")
    return "\n".join(lines)


def compact_line_json(line: ComboLine) -> str:
    """A verified example line as compact JSON with names only (no *_codes)."""
    d = line.to_json()
    d = {k: v for k, v in d.items() if not k.endswith("_codes")}
    for s in d.get("steps", []):
        for k in [k for k in s if k.endswith("_codes")]:
            del s[k]
    return json.dumps(d, separators=(",", ":"))


def plan_prompt(
    deck_name: str, sheet: str, hand_names: list[str], examples: list[ComboLine], candidates: int = 1
) -> str:
    """Ask for one going-first ComboLine as JSON `{"lines":[...]}` for the hand.

    With `candidates` > 1, ask for that many DIFFERENT lines in one call
    (`{"lines": [...]}` with N entries, differing in route or end board).
    """
    hand = "\n".join(f"- {n}" for n in hand_names)
    ex = ""
    if examples:
        shown = "\n\n".join(compact_line_json(e) for e in examples[:3])
        ex = f"\nEXAMPLES (verified {deck_name} lines, names only):\n{shown}\n"
    nb = notes_block(deck_name)
    notes_sec = f"{nb}\n" if nb else ""
    if candidates is not None and candidates > 1:
        ask = (
            f"Write {candidates} DIFFERENT strong turn-1 combo lines this hand allows as JSON:\n"
            f'{{"lines": [ComboLine x {candidates}]}}\n'
            "They must differ in route or end board (different starters, extenders or end pieces) "
            "— not minor reorderings of the same line."
        )
    else:
        ask = (
            "Write the single strongest turn-1 combo line this hand allows as JSON:\n"
            '{"lines": [ComboLine]}'
        )
    return f"""Deck: {deck_name} (going first)

DECKLIST (card sheet):
{sheet}
{notes_sec}
OPENING HAND (exactly these 5 cards):
{hand}
{ex}
{ask}
{SCHEMA_TEXT}

Rules:
- `hand` must be exactly the 5 cards above, in any order.
- Use ONLY cards from the decklist above (main + extra). Every step's card must be in the hand, on the field, in the GY, or reachable at that point via a previous step's search/summon.
- Normal Summon at most once; respect "once per turn" limits.
- Aim for the strongest end board (interruptions) the hand allows.
- `going` must be "first". `end_board` = cards on your field at the end; `end_board_text` = one-line summary; `claimed_interruptions` = opponent-turn interactions or null.
- If the hand cannot combo, return the best simple line (normal summon, set cards, pass) rather than an illegal flashy one.
JSON only, no prose."""


def improve_prompt(
    deck_name: str, sheet: str, hand_names: list[str], old_line_json: str, old_end: list[str]
) -> str:
    """Ask for a strictly stronger turn-1 ComboLine for the same hand.

    The current line is legal and engine-verified; the model must beat it
    (bigger / more resilient end board) or answer `{"lines": []}`.
    """
    hand = "\n".join(f"- {n}" for n in hand_names)
    end = ", ".join(old_end) if old_end else "(empty)"
    nb = notes_block(deck_name)
    notes_sec = f"{nb}\n" if nb else ""
    return f"""Deck: {deck_name} (going first)

DECKLIST (card sheet):
{sheet}
{notes_sec}
OPENING HAND (exactly these 5 cards):
{hand}

CURRENT VERIFIED LINE (legal, compact JSON, names only):
{old_line_json}

CURRENT END BOARD: {end}

This line is legal and verified. Write a strictly stronger turn-1 line for the SAME hand: a bigger or more resilient end board (more interruptions / negates / board presence), using effects the current line leaves unused. Same JSON schema. If no stronger line exists, answer {{"lines": []}}.
Write the single strongest turn-1 combo line this hand allows as JSON:
{{"lines": [ComboLine]}}
{SCHEMA_TEXT}

Rules:
- `hand` must be exactly the 5 cards above, in any order.
- Use ONLY cards from the decklist above (main + extra). Every step's card must be in the hand, on the field, in the GY, or reachable at that point via a previous step's search/summon.
- Normal Summon at most once; respect "once per turn" limits.
- Aim for the strongest end board (interruptions) the hand allows.
- `going` must be "first". `end_board` = cards on your field at the end; `end_board_text` = one-line summary; `claimed_interruptions` = opponent-turn interactions or null.
- If the hand cannot combo, return the best simple line (normal summon, set cards, pass) rather than an illegal flashy one.
JSON only, no prose."""


def repair_prompt(base_prompt: str, line_json: str, verdict_reason: str, explain_text: str) -> str:
    """Base prompt + previous answer + engine rejection + repair instruction."""
    return (
        f"{base_prompt}\n\nYOUR PREVIOUS ANSWER:\n{line_json}\n\n"
        f"The engine rejected it: {verdict_reason}\n{explain_text}\n\n"
        'Return a corrected full line as JSON {"lines": [ComboLine]} (same schema and rules). '
        "JSON only, no prose."
    )


def _board_key(line: ComboLine):
    """Hashable end-board identity: resolved codes when available, else names."""
    codes = line.end_board_codes or []
    resolved = [c for c in codes if c is not None]
    if resolved and len(resolved) == len(codes):
        return ("codes", tuple(sorted(resolved)))
    if resolved:
        return ("mixed", tuple(sorted(resolved)), tuple(sorted(line.end_board or [])))
    return ("names", tuple(sorted(line.end_board or [])))


def _line_score(line: ComboLine, executed_steps: int) -> tuple[int, int]:
    """Best-of-N score: bigger end board wins, ties go to more executed steps."""
    codes = line.end_board_codes
    size = len(codes) if codes is not None else len(line.end_board or [])
    return (size, int(executed_steps))


def _ground_candidate(
    d: dict, hand_names: list[str], deck_name: str, id_tag: str, k: int, attempt: int,
    names: CardNames, suffix: str = "",
) -> ComboLine:
    """Force hand/going/id and ground names to codes (shared by plan/improve)."""
    line = ComboLine.from_json(d)
    line.hand = list(hand_names)
    line.going = "first"
    line.id = f"{deck_name}/plan{id_tag}-h{k}-a{attempt}{suffix}"
    line.deck = deck_name
    line.source_file = "planner"
    # card-less narration steps ("pass", "end turn") are not actions: drop them before grounding
    line.steps = [st for st in line.steps if st.card or st.action not in ("other", "pass")]
    for j, st in enumerate(line.steps):
        st.i = j
    _fill_codes(line, names)
    return line


def select_best(cands: list[tuple]) -> tuple[int, list[tuple]]:
    """Pick the best verified (line, verdict) pair; rest with a distinct end board are extras.

    Score = `len(end_board_codes)`, ties go to more executed steps. Returns
    `(best_index, extras)` where extras differ from the best end board.
    """
    best_i, best_score = -1, None
    for i, (line, v) in enumerate(cands):
        s = _line_score(line, v.executed_steps)
        if best_score is None or s > best_score:
            best_score, best_i = s, i
    if best_i < 0:
        return -1, []
    best_key = _board_key(cands[best_i][0])
    extras = [(ln, vv) for j, (ln, vv) in enumerate(cands) if j != best_i and _board_key(ln) != best_key]
    return best_i, extras


def plan_hands(
    engine,
    deck,
    deck_name: str,
    hands: list[list[int]],
    examples: list[ComboLine],
    names: CardNames,
    complete_many_fn: Callable[[list[str]], list],
    verify_fn: Callable[[list[tuple]], list],
    repairs: int = 2,
    budget: int = 2000,
    budget_long: int = 10000,
    id_tag: str = "",
    candidates: int = 1,
) -> list[dict]:
    """Plan one combo line per hand with up to `repairs` repair rounds.

    Round 0 builds all prompts and calls `complete_many_fn(prompts)` once;
    failed or unparsable hands go to the next round with a repair prompt.
    With `candidates` > 1, round 0 asks for that many different lines per hand
    in ONE call, verifies all of them and keeps the verified one with the best
    score (`len(end_board_codes)`, ties go to more executed steps). Repairs
    apply only when none verified (repairing the first candidate). Other
    verified candidates with a distinct end board are kept in `extras`.
    Returns one result dict per hand:
    `{hand, attempts:[{line, reason, detail, nodes}], ok_line, ok_verdict, extras:[{line, verdict}]}`.
    """
    n_cand = max(1, int(candidates or 1))
    sheet = card_sheet(engine, deck, names)
    hand_names_list = [[names.names.get(c, str(c)) for c in h] for h in hands]
    base_prompts = [plan_prompt(deck_name, sheet, hn, examples, candidates=n_cand) for hn in hand_names_list]
    results: list[dict] = [
        {"hand": h, "attempts": [], "ok_line": None, "ok_verdict": None, "extras": []} for h in hands
    ]
    pending = list(range(len(hands)))
    name_of = names.names
    for attempt in range(repairs + 1):
        if not pending:
            break
        prompts: list[str] = []
        for k in pending:
            if attempt == 0:
                prompts.append(base_prompts[k])
            else:
                if n_cand > 1 and attempt == 1:
                    last = results[k]["attempts"][0]  # repair the first candidate
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
        parsed: dict[int, ComboLine | None] = {}
        parsed_multi: dict[int, list[ComboLine]] = {}
        multi_pos: list[tuple[int, int]] = []  # (hand k, candidate i) per to_verify entry in round 0
        for k, resp in zip(pending, responses):
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
            dicts = _parse_json(resp if isinstance(resp, str) else str(resp))
            if not dicts:
                results[k]["attempts"].append(
                    {
                        "line": None,
                        "line_json": (resp if isinstance(resp, str) else str(resp))[:2000],
                        "reason": "unparsable",
                        "detail": "no ComboLine JSON object found",
                        "nodes": 0,
                        "explain": "",
                    }
                )
                continue
            if attempt == 0 and n_cand > 1:
                # Best-of-N round 0: ground every candidate, verify all.
                got_any = False
                for i, d in enumerate(dicts[:n_cand]):
                    try:
                        line = _ground_candidate(d, hand_names_list[k], deck_name, id_tag, k, attempt, names, suffix=f"-c{i}")
                    except Exception as e:  # noqa: BLE001 - malformed LLM object -> record, try next
                        results[k]["attempts"].append(
                            {
                                "line": None,
                                "line_json": json.dumps(d)[:2000],
                                "reason": "unparsable",
                                "detail": f"malformed line: {e}",
                                "nodes": 0,
                                "explain": "",
                            }
                        )
                        got_any = True
                        continue
                    got_any = True
                    parsed_multi.setdefault(k, []).append(line)
                    b = budget_long if len(line.steps) > 8 else budget
                    to_verify.append((line, deck, b))
                    verify_pos.append(k)
                    multi_pos.append((k, len(parsed_multi[k]) - 1))
                if not got_any:
                    results[k]["attempts"].append(
                        {
                            "line": None,
                            "line_json": (resp if isinstance(resp, str) else str(resp))[:2000],
                            "reason": "unparsable",
                            "detail": "no ComboLine JSON object found",
                            "nodes": 0,
                            "explain": "",
                        }
                    )
                continue
            try:
                line = ComboLine.from_json(dicts[0])
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
            line.hand = list(hand_names_list[k])
            line.going = "first"
            line.id = f"{deck_name}/plan{id_tag}-h{k}-a{attempt}"  # id_tag: e.g. "-s3" (seed), unique across runs
            line.deck = deck_name
            line.source_file = "planner"
            # card-less narration steps ("pass", "end turn") are not actions: drop them before grounding
            line.steps = [st for st in line.steps if st.card or st.action not in ("other", "pass")]
            for j, st in enumerate(line.steps):
                st.i = j
            _fill_codes(line, names)
            parsed[k] = line
            b = budget_long if len(line.steps) > 8 else budget
            to_verify.append((line, deck, b))
            verify_pos.append(k)
        verdicts = verify_fn(to_verify) if to_verify else []
        if attempt == 0 and n_cand > 1:
            # Group (line, verdict) pairs per hand in verify order.
            counters: dict[int, int] = {}
            regrouped: dict[int, list[tuple]] = {}
            for (k, _i), v in zip(multi_pos, verdicts):
                c = counters.get(k, 0)
                counters[k] = c + 1
                line = parsed_multi[k][c]
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
                regrouped.setdefault(k, []).append((line, v))
            for k, pairs in regrouped.items():
                verified = [(ln, vv) for ln, vv in pairs if vv.ok]
                if not verified:
                    continue
                best_i, extras = select_best(verified)
                best_line, best_v = verified[best_i]
                results[k]["ok_line"] = best_line
                results[k]["ok_verdict"] = best_v
                results[k]["extras"] = [{"line": ln, "verdict": vv} for ln, vv in extras]
        else:
            for k, v in zip(verify_pos, verdicts):
                line = parsed[k]
                assert line is not None
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
        # Hands that failed to parse also stay pending (their attempt is recorded above).
        pending = [k for k in pending if results[k]["ok_verdict"] is None]
    return results
