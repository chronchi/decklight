"""Card-name → passcode resolution against BabelCDB.

Never guesses: `resolve` returns [] when the name cannot be matched
confidently (the caller flags it in `unresolved_names`).

Nickname layer: `resolve_many_in_pool` maps nicknames/abbreviations
("Albaz", "Mirrorjade", "Aluber") to exactly one candidate card name
from a constrained pool (the deck's decklist ∪ the line's other
resolved cards). The LLM must answer with the exact candidate string
or null; anything else is rejected.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from ygo_zero.engine.resources import default_cdbs

logger = logging.getLogger(__name__)

NICKNAME_SYSTEM = (
    "You map Yu-Gi-Oh! card nicknames/abbreviations to full card names. Reply with JSON only, no prose."
)


def normalize(name: str) -> str:
    """Case/punctuation/quote-insensitive key: lowercase alphanumeric only."""
    n = name.lower()
    # common unicode punctuation → ascii before stripping
    n = n.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]", "", n)


class CardNames:
    def __init__(self, cdb_paths: list[Path] | None = None) -> None:
        self.names: dict[int, str] = {}  # code -> name
        self.alias: dict[int, int] = {}  # code -> alias (0 = canonical)
        for path in cdb_paths if cdb_paths is not None else default_cdbs():
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                for code, name in con.execute("SELECT id, name FROM texts").fetchall():
                    self.names[code] = name
                for code, alias in con.execute("SELECT id, alias FROM datas").fetchall():
                    self.alias[code] = alias
            finally:
                con.close()
        # normalized name -> codes (later cdbs already overrode earlier ones)
        self._index: dict[str, list[int]] = {}
        for code, name in self.names.items():
            self._index.setdefault(normalize(name), []).append(code)
        self._canon = {
            n: sorted(cs, key=lambda c: (self.alias.get(c, 0) != 0, c)) for n, cs in self._index.items()
        }
        self._keys = list(self._index)

    def resolve(self, name: str) -> list[int]:
        """Return matching codes, best first; [] if unresolved. Never guesses."""
        key = normalize(name)
        if not key:
            return []
        # 1. exact (case/punctuation/quote-insensitive), canonical (alias == 0) first
        if key in self._canon:
            return list(self._canon[key])
        # 2. conservative fuzzy: unique match with ratio >= 0.92 only
        scored = [(difflib.SequenceMatcher(None, key, k).ratio(), k) for k in self._keys]
        best = [k for r, k in scored if r >= 0.92]
        if len(best) == 1:
            return list(self._canon[best[0]])
        return []

    def candidate_names(self, pool_codes: set[int]) -> list[str]:
        """Full card names for pool codes, sorted, deduplicated."""
        return sorted({self.names[c] for c in pool_codes if c in self.names})

    def resolve_in_pool(
        self, name: str, pool_codes: set[int], complete_fn: Callable[..., str] | None = None
    ) -> int | None:
        """Resolve `name`, constraining any guess to `pool_codes`.

        Plain `resolve` first (exact/fuzzy, pool-independent). If that fails,
        fall back to one LLM nickname call constrained to the pool's names
        (callers with many names should prefer the batched
        `resolve_many_in_pool`: one call per guide). Returns None when
        unresolvable; never guesses outside the pool.
        """
        hit = self.resolve(name)
        if hit:
            return hit[0]
        mapping = resolve_many_in_pool(self, [name], pool_codes, complete_fn=complete_fn)
        return mapping.get(name)


def _nickname_prompt(unresolved: list[str], candidates: list[str]) -> str:
    lines = "\n".join(f"- {n}" for n in unresolved)
    cands = "\n".join(f"- {c}" for c in candidates)
    return f"""Map each Yu-Gi-Oh! card nickname/abbreviation below to exactly one card from the candidate list.

NICKNAMES:
{lines}

CANDIDATES (reply with these strings exactly, character-for-character):
{cands}

Rules:
- Answer with a JSON object mapping each nickname to exactly one candidate string, or null if none matches.
- Copy the candidate string EXACTLY (same spelling, punctuation, spacing). Any value that is not an exact candidate string is rejected.
- Example: {{"Albaz": "Fallen of Albaz", "Mirrorjade": "Mirrorjade the Iceblade Dragon", "Unknown Xyz": null}}
JSON only, no prose."""


def _parse_nickname_response(text: str, candidates: set[str]) -> dict[str, str | None]:
    """Parse the LLM nickname mapping; keep only exact-candidate values."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, str | None] = {}
    for k, v in obj.items():
        if v is None:
            out[str(k)] = None
        elif isinstance(v, str) and v in candidates:
            out[str(k)] = v
        else:
            logger.warning("nickname: rejecting non-candidate mapping %r -> %r", k, v)
            out[str(k)] = None
    return out


def resolve_many_in_pool(
    names: CardNames,
    unresolved: list[str],
    pool_codes: set[int],
    complete_fn: Callable[..., str] | None = None,
) -> dict[str, int | None]:
    """Batch-map `unresolved` nicknames to pool codes with one LLM call.

    Returns {nickname: code | None}. Names that plain-`resolve` need no LLM
    and are answered directly. The LLM must answer with exact candidate
    strings; anything else becomes None. Empty input or an empty pool costs
    no LLM call. `complete_fn` defaults to `llm.complete` (disk-cached);
    pass a stub in tests. Any LLM error maps every still-unresolved name
    to None (never guess).
    """
    result: dict[str, int | None] = {}
    candidates = names.candidate_names(pool_codes)
    need: list[str] = []
    for n in unresolved:
        hit = names.resolve(n)
        if hit:
            result[n] = hit[0]
        elif n not in result:
            result[n] = None
            need.append(n)
    need = [n for n in need if result[n] is None]
    if not need or not candidates:
        return result
    if complete_fn is None:
        from ygo_zero.demos import llm as _llm

        complete_fn = _llm.complete
    try:
        response = complete_fn(
            _nickname_prompt(need, candidates),
            system=NICKNAME_SYSTEM,
        )
    except Exception as e:  # noqa: BLE001 - LLM failure must not kill extraction
        logger.warning("nickname: LLM call failed: %s", e)
        return result
    cand_set = set(candidates)
    name_to_code = {names.names[c]: c for c in pool_codes if c in names.names}
    for nick, full in _parse_nickname_response(response, cand_set).items():
        if full is not None and nick in result:
            code = name_to_code.get(full)
            result[nick] = code if code in pool_codes else None
    return result
