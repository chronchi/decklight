"""Audit pool decks for missing namesake/key cards.

For every `decks/pool/*.ydk` (skipping `*-meta*`), find card names that are quoted (`"Name"`) in the
texts of >=3 other deck cards but are themselves absent from the deck — e.g. dark-magician.ydk lacking
"Dark Magician" (46986414), which deadens Navigation, Circle, Souls' 2nd mode, Eye of Timaeus. Only
names that are real cards (in the card DB) are flagged, so archetype quotes like `"Tenyi"` are ignored.

Prints a table: deck, missing card, how many deck cards mention it, and which ones.

    uv run python scripts/check_decks.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.decks import load_ydk
from ygo_zero.demos.game import _effect_text
from ygo_zero.demos.names import CardNames

_QUOTED = re.compile(r'"([^"]+)"')
MIN_MENTIONS = 3


def audit(deck_path: Path, names: CardNames) -> list[tuple[str, int, list[str]]]:
    """Missing key cards of one deck as (name, mention count, mentioning cards) rows."""
    deck = load_ydk(deck_path)
    codes = list(deck.main) + list(deck.extra)
    have = {names.names.get(c, "") for c in codes} - {""}
    texts = {c: _effect_text(c) for c in set(codes)}
    by_name: dict[str, set[str]] = {}  # quoted card name -> deck cards mentioning it
    for c in set(codes):
        me = names.names.get(c, "")
        for q in _QUOTED.findall(texts.get(c, "")):
            if q != me:
                by_name.setdefault(q, set()).add(me or f"#{c}")
    real = set(names.names.values())
    out = []
    for q, mentioners in sorted(by_name.items()):
        if len(mentioners) >= MIN_MENTIONS and q in real and q not in have:
            out.append((q, len(mentioners), sorted(mentioners)[:5]))
    return out


def main() -> None:
    names = CardNames()
    decks = sorted((ROOT / "decks" / "pool").glob("*.ydk"))
    decks = [d for d in decks if "-meta" not in d.name]
    print(f"{'deck':28} {'missing card':38} {'n':>3}  mentioned in")
    print("-" * 110)
    flagged = 0
    for deck_path in decks:
        for q, n, who in audit(deck_path, names):
            flagged += 1
            print(f"{deck_path.stem:28} {q:38} {n:>3}  {', '.join(who)}")
    if not flagged:
        print("(no missing key cards)")


if __name__ == "__main__":
    main()
