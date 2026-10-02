"""Card-name → passcode resolution against BabelCDB.

Never guesses: `resolve` returns [] when the name cannot be matched
confidently (the caller flags it in `unresolved_names`).
"""

from __future__ import annotations

import difflib
import re
import sqlite3
from pathlib import Path

from ygo_zero.engine.resources import default_cdbs


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
