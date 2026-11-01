"""Per-deck summary for the lobby deck browser (web/render/decks.js).

`scripts/build_deck_index.py` writes deck_index.json for the lobby's folder decks; the server adds entries for the
decks it learns at run time (fine-tunes' own decks, decks pasted this session) with `deck_summary`, so every deck in
the browser has its card list (user 2026-11-01: "the fine tuned decks in the search do not have the deck list or the
cards").
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

INDEX_PATH = Path(__file__).resolve().parent / "deck_index.json"
# A summon type counts as the deck's own when it is more than a staple splash (measured on the 189 lobby decks).
MECH_MIN = {"Fusion": 4, "Synchro": 4, "Xyz": 4, "Link": 6, "Ritual": 2, "Pendulum": 4}


def deck_summary(main: list[int], extra: list[int], cards: dict, staple: set[int] | frozenset[int] = frozenset(),
                 key_cards: list[int] | None = None) -> tuple[dict, int]:
    """(entry without the text fields, number of main-deck codes missing from the card DB)."""
    from decklight.engine import constants as C

    def ty(c: int) -> int:
        return cards[c].type if c in cards else 0

    cnt: collections.Counter = collections.Counter()
    for c in main:
        t = ty(c)
        cnt["m" if t & C.TYPE_MONSTER else "s" if t & C.TYPE_SPELL else "t" if t & C.TYPE_TRAP else "?"] += 1
        if t & C.TYPE_MONSTER and t & C.TYPE_RITUAL:
            cnt["Ritual"] += 1
        if t & C.TYPE_PENDULUM:
            cnt["Pendulum"] += 1
    for c in extra:
        t = ty(c)
        for name, bit in (("Fusion", C.TYPE_FUSION), ("Synchro", C.TYPE_SYNCHRO), ("Xyz", C.TYPE_XYZ),
                          ("Link", C.TYPE_LINK)):
            if t & bit:
                cnt[name] += 1
    sig = list((key_cards or [])[:3])
    if len(sig) < 3:  # no primer: the most-copied distinctive monsters
        mons = collections.Counter(c for c in main if c not in staple and ty(c) & C.TYPE_MONSTER)
        sig += [c for c, _ in mons.most_common() if c not in sig][: 3 - len(sig)]
    return {"mech": [m for m, lo in MECH_MIN.items() if cnt[m] >= lo], "counts": [cnt["m"], cnt["s"], cnt["t"]],
            "sig": sig, "main": [[c, x] for c, x in collections.Counter(main).items()],
            "extra": [[c, x] for c, x in collections.Counter(extra).items()]}, cnt["?"]


class DeckIndex:
    """The committed index plus run-time decks. `json_for(extra_decks)` is what /deck_index.json serves."""

    def __init__(self, path: Path = INDEX_PATH):
        self.base = json.loads(path.read_text())  # raises: the server warns once and serves no index
        self.staple = frozenset(self.base.get("staples", []))
        self._cache: dict[tuple, dict] = {}
        self._cards: dict | None = None

    def entry(self, key: str, name: str, origin: str, main: list[int], extra: list[int]) -> tuple[dict, dict]:
        """(entry, names of its cards); cached by content."""
        ck = (key, name, origin, tuple(main), tuple(extra))
        if ck not in self._cache:
            from decklight.engine.resources import load_cards
            from decklight.web.view import DB

            if self._cards is None:
                self._cards = load_cards()
            e, _missing = deck_summary(main, extra, self._cards, self.staple)
            e.update({"arch": "", "style": "", "about": "", "origin": origin, "year": None})  # archetype unknown
            names = DB.names()
            self._cache[ck] = (e, {str(c): names.get(c, f"#{c}") for c in set(main) | set(extra)})
        return self._cache[ck]

    def json_for(self, extra_decks: list[tuple[str, str, str, list[int], list[int]]]) -> dict:
        if not extra_decks:
            return self.base
        decks, names = dict(self.base["decks"]), dict(self.base["names"])
        for k, name, origin, main, extra in extra_decks:
            e, n = self.entry(k, name, origin, main, extra)
            decks[k] = e
            names.update(n)
        return {**self.base, "decks": decks, "names": names}
