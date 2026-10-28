#!/usr/bin/env python3
"""Card codes for the Rust core's `YGO_CARDS_ONLY` build filter.

Prints the space-separated `c<code>` tokens of every card in
`decks/{pool,gauntlet,newcards,holdout,variants,x4b}/**/*.ydk` (main + extra + side),
plus each card's alias target from the card DB, deduplicated and sorted.

The format matches `third_party/ygopro-core-rust/ygo-cards/build.rs`
(`YGO_CARDS_ONLY="c12345 c67890"`: tokens split on non-digits, so the `c`
prefix is what keeps tokens separate). Used by `scripts/build_rust_core.sh`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks.ydk import parse_ydk
from decklight.engine.resources import load_cards

DIRS = ("pool", "gauntlet", "newcards", "holdout", "variants", "x4b")


def main() -> None:
    codes: set[int] = set()
    for d in DIRS:
        for path in sorted((ROOT / "decks" / d).rglob("*.ydk")):
            deck = parse_ydk(path.read_text(encoding="utf-8"))
            codes.update(deck.main)
            codes.update(deck.extra)
            codes.update(deck.side)
    codes.discard(0)
    cards = load_cards()
    for code in list(codes):
        alias = cards[code].alias if code in cards else 0
        if alias:
            codes.add(alias)
    print(" ".join(f"c{c}" for c in sorted(codes)))


if __name__ == "__main__":
    main()
