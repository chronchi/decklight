"""Stage 2 (ADR 0002): guide text → structured ComboLine objects.

Pipeline per guide file: split into combo sections → one LLM call per
section asking for strict JSON (card *names* only) → validate against the
schema → resolve names to codes via names.py → flag off-decklist cards.
One ComboLine per distinct combo route.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

from decklight.demos import llm as _llm
from decklight.demos.names import CardNames, resolve_global_unique, resolve_many_in_pool
from decklight.demos.schema import ACTIONS, ComboLine

logger = logging.getLogger(__name__)

# Independent per-section LLM calls for one guide run concurrently.
EXTRACT_WORKERS = 6


def _complete_sections(
    complete_fn: Callable[..., str],
    prompts: list[str],
    max_workers: int = EXTRACT_WORKERS,
) -> list[str | BaseException]:
    """Complete section prompts concurrently, results in input (serial) order.

    The full prompt list is built before anything is submitted, so a caller
    that budgets calls (scripts/extract_lines.py --max-calls) sees exactly
    the same call count as the serial version. Per-section failures are
    returned in place so one bad section never kills the file.
    """
    if complete_fn is _llm.complete:
        return _llm.complete_many(prompts, system=SYSTEM, max_workers=max_workers, return_exceptions=True)
    if len(prompts) <= 1 or max_workers <= 1:
        out: list[str | BaseException] = []
        for p in prompts:
            try:
                out.append(complete_fn(p, system=SYSTEM))
            except Exception as e:  # noqa: BLE001 - isolated per section, as in serial
                out.append(e)
        return out
    with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as ex:
        futs = [ex.submit(complete_fn, p, system=SYSTEM) for p in prompts]
        res: list[str | BaseException] = []
        for f in futs:
            try:
                res.append(f.result())
            except Exception as e:  # noqa: BLE001 - isolated per section, as in serial
                res.append(e)
        return res


SYSTEM = (
    "You are a Yu-Gi-Oh! combo analyst. Extract every distinct combo route "
    "from the guide section into strict JSON. Use exact card names as written. "
    "Never invent card IDs — output names only. Reply with JSON only, no prose."
)

# Lines that usually start a new combo route/section.
_SECTION_RE = re.compile(
    r"(?im)^[^\S\n]*(?:#{1,4}\s+.*(?:combo|route|line|opening|setup).*"
    r"|COMBO\s*\d+.*|(?:combo|route|line)\s*\d+\s*[:.\-–—].*"
    r"|(?:starting\s+with|opening)\s*[:.\-–—].*)$"
)
# Fetched-guide header ends at a rule of '=' chars; metadata above it.
_RULE_RE = re.compile(r"(?m)^={10,}\s*$")
_URL_RE = re.compile(r"(?m)^URL:\s*(\S+)")
_MAX_CHUNK = 8000


def split_header(text: str) -> tuple[dict, str]:
    """Split a fetched-guide file into (metadata, body)."""
    meta: dict = {}
    m = _RULE_RE.search(text)
    body = text[m.end() :] if m else text
    head = text[: m.start()] if m else ""
    u = _URL_RE.search(head)
    if u:
        meta["url"] = u.group(1)
    return meta, body.strip()


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split guide body into (title, text) combo sections."""
    bounds = [m.start() for m in _SECTION_RE.finditer(body)]
    if not bounds:
        return [("full", body.strip())] if body.strip() else []
    bounds.append(len(body))
    out = []
    for a, b in pairwise(bounds):
        chunk = body[a:b].strip()
        if len(re.sub(r"\W", "", chunk)) < 8:
            continue
        title = chunk.splitlines()[0].strip("# *-")[:100]
        out.append((title, chunk))
    if not out and body.strip():
        return [("full", body.strip())]
    # hard-split overlong sections at blank lines so one call stays small
    final = []
    for title, chunk in out:
        while len(chunk) > _MAX_CHUNK:
            cut = chunk.rfind("\n\n", 0, _MAX_CHUNK)
            cut = cut if cut > 0 else _MAX_CHUNK
            final.append((title, chunk[:cut]))
            chunk = chunk[cut:].strip()
        final.append((title, chunk))
    return final


# --- Guide decklists (regex, no LLM) -------------------------------------------
# Guides sometimes print their own decklist (Game8 icon rows, "3x Name" runs,
# "- Name x3" bullets). When found it grounds verification better than our
# fixtures, so extract it into data/demos/decklists/<deck>/<source>.json.

_DECK_SECTION_RE = re.compile(r"(?i)^(?:#{1,4}\s*)?(main deck|extra deck|side deck)\b(.*)$")
_EXTRA_PREFIX_RE = re.compile(r"(?i)^(EXTRA|SIDE)\s*:\s*(.*)$")
_ICON_ALTS_RE = re.compile(r"!\[([^\]]+)\]")
_QTY_TRAIL_RE = re.compile(r"\*{0,2}[xX](\d+)\*{0,2}\s*$")
_LEAD_QTY_RE = re.compile(r"^\s*(?:[-•]\s*)?(\d+)\s*[xX]\s+(.+?)\s*$")


def _decklist_entry(line: str) -> tuple[str, int] | None:
    """Parse one decklist line into (card name, count)."""
    s = line.strip()
    if not s:
        return None
    # Game8 icon rows: [![Name Icon](...)](url)**xN**
    alts = _ICON_ALTS_RE.findall(s)
    q = _QTY_TRAIL_RE.search(s)
    if alts and q:
        name = alts[-1]
        name = re.sub(r"\s+Icon\s*$", "", name).strip()
        return (name, int(q.group(1))) if name else None
    # "- Name xN" bullets (also catches a trailing **xN** without icons)
    if q:
        body = _QTY_TRAIL_RE.sub("", s).strip()
        body = re.sub(r"^[-•]\s*", "", body).strip()
        # skip prose lines ("...the Deck...") — bullets with 'xN' suffix only
        if body and not re.search(r"[.!?]\s", body) and len(body) <= 60:
            return (body, int(q.group(1)))
        return None
    # "Nx Name" runs (ygoprodeck inline lists, bullets); skip bare numbers/prose
    m = _LEAD_QTY_RE.match(s)
    if m:
        count, rest = int(m.group(1)), m.group(2).strip().rstrip(",;")
        rest = re.sub(r"\s*\(see page.*$", "", rest).strip()
        if 1 <= count <= 3 and rest and len(rest) <= 60 and not re.search(r"[.!?]\s", rest):
            # avoid combo prose ("2x [[Spirit Dragon]] to Special Summon...")
            if re.search(r"\bto\b\s+\w+\s+\w+$", rest):
                return None
            return (rest, count)
    return None


def extract_guide_decklist(text: str, names: CardNames) -> dict | None:
    """Extract a {main, extra, unresolved} decklist from guide text, if any.

    Only lines inside Main/Extra Deck sections are considered (so combo prose
    like "Special Summon 2x ..." is never picked up). Returns None when no
    usable decklist is found. Codes are expanded by count (3x = 3 entries).
    """
    main: list[int] = []
    extra: list[int] = []
    unresolved: list[str] = []
    section: str | None = None
    seen_sections = 0
    for raw in text.splitlines():
        line = raw.strip()
        ms = _DECK_SECTION_RE.match(line)
        if ms:
            kind, tail = ms.group(1).lower(), ms.group(2).strip()
            section = "extra" if kind.startswith("extra") else "main" if kind.startswith("main") else None
            seen_sections += 1
            # "MAIN DECK (45, ...): 3x A, 3x B, ..." — inline list on the heading
            tail = re.sub(r"^[\s:(\[]+", "", tail)
            if tail and section:
                for chunk in re.split(r",\s*", tail):
                    e = _decklist_entry(chunk)
                    if e:
                        _add_decklist_entry(e, names, main if section == "main" else extra, unresolved)
            continue
        me = _EXTRA_PREFIX_RE.match(line)
        if me:
            section = "extra"
            seen_sections += 1
            for chunk in re.split(r",\s*", me.group(2)):
                e = _decklist_entry(chunk)
                if e:
                    _add_decklist_entry(e, names, extra, unresolved)
            continue
        if line.startswith("#"):
            section = None
            continue
        if section in ("main", "extra"):
            # inline comma runs ("3x A, 3x B, ...") as well as one-per-line
            chunks = re.split(r",\s*", line) if re.match(r"^\s*(\d+\s*[xX]\s+|[-•]\s*)", line) else [line]
            for chunk in chunks:
                e = _decklist_entry(chunk)
                if e:
                    _add_decklist_entry(e, names, main if section == "main" else extra, unresolved)
    if seen_sections == 0 or len(main) < 5:
        return None
    return {"main": main, "extra": extra, "unresolved": sorted(set(unresolved))}


def _add_decklist_entry(
    entry: tuple[str, int], names: CardNames, out: list[int], unresolved: list[str]
) -> None:
    name, count = entry
    hit = names.resolve(name)
    if not hit:
        if name not in unresolved:
            unresolved.append(name)
        return
    out.extend([hit[0]] * max(1, min(count, 3)))


def detect_format(text: str) -> str:
    t = text.lower()
    md = "master duel" in t or re.search(r"(?m)^(?:format|site)\s*:.*\bmd\b", t) is not None
    ocg = "ocg" in t
    tcg = "tcg" in t
    if md and not (ocg or tcg):
        return "MD"
    if tcg and not ocg:
        return "TCG"
    if ocg and not tcg:
        return "OCG"
    return "unknown"


def _prompt(section_title: str, section_text: str, deck: str) -> str:
    actions = ", ".join(ACTIONS)
    return f"""Deck: {deck}
Guide section titled: {section_title}

GUIDE TEXT:
{section_text}

Extract every distinct combo route in this section as JSON:
{{"lines": [ComboLine, ...]}}
One object per distinct combo route (different starter or different end board = different route).
Each ComboLine has: title, hand (starting-hand card names), requires (other cards the line needs in deck/extra), steps (ordered; each step: i (0-based), action (one of: {actions}), card, effect (free-text hint), materials, targets, results (name lists), text (original sentence)), end_board (card names on field), end_board_text, claimed_interruptions (or null), going (first|second|unknown).
going = "first" for turn-1 setup combos (the default: the combo builds its board from its own hand with no opponent cards needed); "second" ONLY if the combo requires the opponent to already have cards on the field or a Battle Phase; otherwise "unknown". Conditional removal ("if the opponent has a monster, destroy it") does NOT make a combo going-second.
Output card NAMES only; omit all *_codes fields (filled automatically). A step's card may be "" with action "other" for game actions without a card (e.g. "send equipped card to GY").

CRITICAL — results attribution: a step's `results` = ONLY the cards that THIS step itself puts into
the hand, field, GY or banished pile when it resolves. Cards that arrive in LATER steps (End Phase
triggers, follow-up searches/summons) belong to those later steps — NEVER copy them into the
current step's results. Likewise `materials` = only cards consumed AS COST/MATERIAL by this step.

Few-shot examples (from a Blue-Eyes guide; note how the End Phase arrival stays in its own step):
Guide: "Activate Sage with Eyes of Blue to send The White Stone of Ancients and itself to the Graveyard, to Special Summon Dragon Spirit of White from the Deck."
Step: {{"i": 1, "action": "activate", "card": "Sage with Eyes of Blue", "effect": "send White Stone + itself; Special Summon Dragon Spirit of White", "materials": ["The White Stone of Ancients", "Sage with Eyes of Blue"], "targets": [], "results": ["Dragon Spirit of White"], "text": "Activate Sage with Eyes of Blue to send The White Stone of Ancients and itself to the Graveyard, to Special Summon Dragon Spirit of White from the Deck."}}
Guide: "At the End Phase, the effect of The White Stone of Ancients will trigger, and Special Summon Blue-Eyes Jet Dragon from the Deck."
Step: {{"i": 4, "action": "special_summon", "card": "The White Stone of Ancients", "effect": "End Phase trigger; Special Summon Jet Dragon", "materials": [], "targets": [], "results": ["Blue-Eyes Jet Dragon"], "text": "At the End Phase, the effect of The White Stone of Ancients will trigger, and Special Summon Blue-Eyes Jet Dragon from the Deck."}}
(Wrong: putting "Blue-Eyes Jet Dragon" in the Sage step's results — it only arrives at the End Phase step.)
JSON only, no prose."""


def _parse_json(text: str) -> list[dict]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        lines = obj.get("lines", [])
        return [x for x in lines if isinstance(x, dict)] if isinstance(lines, list) else []
    return []


def _fill_codes(line: ComboLine, names: CardNames) -> None:
    unresolved: list[str] = []

    def code(name: str) -> int | None:
        if not name:
            return None
        hit = names.resolve(name)
        if not hit:
            if name not in unresolved:
                unresolved.append(name)
            return None
        return hit[0]

    line.hand_codes = [code(n) for n in line.hand]
    line.requires_codes = [code(n) for n in line.requires]
    line.end_board_codes = [code(n) for n in line.end_board]
    for s in line.steps:
        s.card_code = code(s.card) if s.card else None
        # step card already recorded via code(); avoid double-adding to unresolved
        s.materials_codes = [code(n) for n in s.materials]
        s.targets_codes = [code(n) for n in s.targets]
        s.results_codes = [code(n) for n in s.results]
    # drop the "" / step-card placeholders that code() may have skipped; keep real ones
    line.unresolved_names = unresolved


def extract_lines(
    guide_path: str | Path,
    deck: str,
    decklist_codes: set[int] | None = None,
    names: CardNames | None = None,
    complete_fn: Callable[..., str] | None = None,
) -> list[ComboLine]:
    """Extract structured combo lines from one guide file."""
    guide_path = Path(guide_path)
    raw = guide_path.read_text(encoding="utf-8", errors="replace")
    meta, body = split_header(raw)
    names = names if names is not None else CardNames()
    complete_fn = complete_fn if complete_fn is not None else _llm.complete
    fmt = detect_format(raw)
    sections = split_sections(body)
    prompts = [_prompt(title, chunk, deck) for title, chunk in sections]
    responses = _complete_sections(complete_fn, prompts)
    lines: list[ComboLine] = []
    idx = 0
    for (title, chunk), response in zip(sections, responses):
        if isinstance(response, BaseException):
            logger.warning("extract_lines: skipping section %r: %s", title, response)
            continue
        for d in _parse_json(response):
            try:
                line = ComboLine.from_json(d)
            except Exception as e:  # noqa: BLE001 - skip malformed LLM objects
                logger.warning("extract_lines: skipping malformed line: %s", e)
                continue
            line.id = f"{deck}/{guide_path.stem}:{idx}"
            line.deck = deck
            line.source_file = guide_path.name
            line.source_url = meta.get("url", "")
            line.format = fmt
            if not line.title:
                line.title = title[:100]
            if line.going == "unknown":
                t = (title + " " + chunk[:500]).lower()
                if "going second" in t:
                    line.going = "second"
                elif "going first" in t:
                    line.going = "first"
            if not line.steps:
                # The verifier rejects 0-step lines as vacuous; drop them here.
                # idx is still consumed so surviving lines keep stable ids (with
                # gaps where dropped lines were).
                logger.info("extract_lines: dropping 0-step line %r from %s", line.title, guide_path.name)
                idx += 1
                continue
            _fill_codes(line, names)
            if decklist_codes is not None:
                outside = sorted(
                    {
                        n
                        for n, c in zip(line.hand + line.requires, line.hand_codes + line.requires_codes)
                        if c is not None and c not in decklist_codes
                    }
                )
                if outside:
                    extra = f"uses cards outside decklist: {outside}"
                    line.notes = f"{line.notes} | {extra}" if line.notes else extra
            lines.append(line)
            idx += 1
    _resolve_nicknames(lines, raw, decklist_codes, names, complete_fn)
    return lines


def _resolve_nicknames(
    lines: list[ComboLine],
    guide_text: str,
    decklist_codes: set[int] | None,
    names: CardNames,
    complete_fn: Callable[..., str],
) -> None:
    """One batched LLM call mapping a guide's unresolved nicknames into the pool,
    then a deterministic whole-DB fallback for what the pool cannot map.

    Pool = the deck's decklist codes ∪ the guide's own decklist (if any) ∪
    the guide's already-resolved codes. Only exact candidate strings from the
    LLM are accepted (`resolve_many_in_pool` enforces that). Patched codes
    are removed from each line's `unresolved_names`.

    Fallback (`resolve_global_unique`, no LLM calls): a still-unresolved
    nickname that uniquely matches one card name in the whole DB is patched
    the same way, and — because it is by definition outside the pool deck —
    appended to the line's `requires` (with a `global_fallback:` note tag) so
    the verifier's augmented-deck pass adds it to the deck. Ambiguous or
    generic nicknames stay unresolved.
    """
    need = sorted({n for line in lines for n in line.unresolved_names if n})
    if not need:
        return
    pool: set[int] = set(decklist_codes) if decklist_codes else set()
    try:
        guide_dl = extract_guide_decklist(guide_text, names)
    except Exception as e:  # noqa: BLE001 - decklist parse must not kill extraction
        logger.warning("extract_lines: guide decklist parse failed: %s", e)
        guide_dl = None
    if guide_dl:
        pool |= set(guide_dl["main"]) | set(guide_dl["extra"])
    for line in lines:
        pool |= {c for c in line.hand_codes + line.requires_codes + line.end_board_codes if c is not None}
        for s in line.steps:
            if s.card_code is not None:
                pool.add(s.card_code)
            pool |= {c for c in s.materials_codes + s.targets_codes + s.results_codes if c is not None}
    if not pool:
        return
    mapping = resolve_many_in_pool(names, need, pool, complete_fn=complete_fn)
    fixed = {n: c for n, c in mapping.items() if c is not None}
    global_fixed: dict[str, tuple[int, str]] = {}
    # Tiebreak context for ambiguous nicknames: the guide's resolved names
    # plus the pool deck's names (the pool already unions both).
    context_names = names.candidate_names(pool)
    for n in need:
        if n in fixed:
            continue
        hit = resolve_global_unique(names, n, context_names)
        if hit is not None:
            global_fixed[n] = hit
    if global_fixed:
        logger.info(
            "extract_lines: global fallback resolved %d nicknames: %s",
            len(global_fixed),
            sorted(global_fixed),
        )
    if not fixed and not global_fixed:
        return
    all_fixed = {**fixed, **{n: c for n, (c, _) in global_fixed.items()}}
    for line in lines:
        had = set(line.unresolved_names)
        _patch_codes(line, all_fixed)
        line.unresolved_names = [n for n in line.unresolved_names if n not in all_fixed]
        for n in sorted(had):
            if n not in global_fixed:
                continue
            code, full = global_fixed[n]
            tag = f"global_fallback: {n}->{full}"
            line.notes = f"{line.notes} | {tag}" if line.notes else tag
            if code in line.hand_codes or code in line.requires_codes:
                continue
            line.requires.append(full)
            line.requires_codes.append(code)


def _patch_codes(line: ComboLine, fixed: dict[str, int]) -> None:
    """Fill None codes whose name the nickname layer resolved."""

    def fix(names_list: list[str], codes_list: list[int | None]) -> None:
        for j, (n, c) in enumerate(zip(names_list, codes_list)):
            if c is None and n in fixed:
                codes_list[j] = fixed[n]

    fix(line.hand, line.hand_codes)
    fix(line.requires, line.requires_codes)
    fix(line.end_board, line.end_board_codes)
    for s in line.steps:
        if s.card_code is None and s.card in fixed:
            s.card_code = fixed[s.card]
        fix(s.materials, s.materials_codes)
        fix(s.targets, s.targets_codes)
        fix(s.results, s.results_codes)


def step_card_sequence(line: ComboLine) -> list[str]:
    """Normalized card-name sequence of a line's steps (for overlap scoring)."""
    from decklight.demos.names import normalize

    return [normalize(s.card) for s in line.steps if s.card]


def sequence_overlap(a: list[str], b: list[str]) -> float:
    """Multiset F1 overlap of two card-name sequences."""
    if not a or not b:
        return 0.0
    ca: dict[str, int] = {}
    for x in a:
        ca[x] = ca.get(x, 0) + 1
    inter = 0
    for x in b:
        if ca.get(x, 0):
            ca[x] -= 1
            inter += 1
    return 2 * inter / (len(a) + len(b))
