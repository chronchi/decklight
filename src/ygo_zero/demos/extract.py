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
from itertools import pairwise
from pathlib import Path

from ygo_zero.demos import llm as _llm
from ygo_zero.demos.names import CardNames
from ygo_zero.demos.schema import ACTIONS, ComboLine

logger = logging.getLogger(__name__)

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
Output card NAMES only; omit all *_codes fields (filled automatically). A step's card may be "" with action "other" for game actions without a card (e.g. "send equipped card to GY").
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
    lines: list[ComboLine] = []
    idx = 0
    for title, chunk in split_sections(body):
        try:
            response = complete_fn(_prompt(title, chunk, deck), system=SYSTEM)
        except Exception as e:  # noqa: BLE001 - one bad section must not kill the file
            logger.warning("extract_lines: skipping section %r: %s", title, e)
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
    return lines


def step_card_sequence(line: ComboLine) -> list[str]:
    """Normalized card-name sequence of a line's steps (for overlap scoring)."""
    from ygo_zero.demos.names import normalize

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
