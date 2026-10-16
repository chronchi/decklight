"""Slot-swap tech-card helpers: same-position card swaps, opening exposure, card names.

A same-length deck with a card replaced **in the same list position** gets the
bit-identical shuffle permutation as the original: `shuffle_main`
(`engine/duel.py`) seeds only on (seed, team) and loops over `len(main)`, so
every deal is identical except for the swapped card's identity. `.ydk` loading
(`decks/ydk.py`) preserves list order, and `swap` below replaces in place, so
the swap preserves positions end to end.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from pathlib import Path

from ygo_zero.decks.ydk import Deck
from ygo_zero.demos.names import CardNames
from ygo_zero.engine.duel import shuffle_main

HAND_SIZE = 5  # DuelConfig.hand; the player going first draws no card on turn 1 (core rule)


@cache
def _names() -> CardNames:
    return CardNames()


def card_name(code: int) -> str:
    """English name of `code` from the card DB; raises KeyError when unknown."""
    name = _names().names.get(int(code))
    if name is None:
        raise KeyError(f"unknown card code {code}")
    return name


def find_code(name: str) -> int:
    """Passcode for a card name (exact, punctuation-insensitive; canonical art first).

    Raises ValueError when the name does not resolve to exactly one card.
    """
    hit = _names().resolve(name)
    if not hit:
        raise ValueError(f"card name {name!r} does not resolve against the card DB")
    return hit[0]


def resolve_code(name_or_code: str | int) -> int:
    """A name (via `find_code`) or a passcode (int or numeric string) to a passcode."""
    if isinstance(name_or_code, int):
        return name_or_code
    s = str(name_or_code).strip()
    if s.isdigit():
        return int(s)
    return find_code(s)


def swap(deck: Deck, out_code: int, in_code: int, n: int | None = None) -> Deck:
    """New deck with the first `n` (default all) copies of `out_code` replaced in place by `in_code`.

    The swap is positional: every other list position is untouched, so the deck
    keeps its length and order and shuffles identically except at the swapped
    slots. `out_code` must be in main or extra (not both: ambiguous); `n` must
    not exceed the available copies.
    """
    out_code, in_code = int(out_code), int(in_code)
    in_main = out_code in deck.main
    in_extra = out_code in deck.extra
    if in_main and in_extra:
        raise ValueError(f"code {out_code} is in both main and extra: disambiguate the swap")
    if not in_main and not in_extra:
        raise ValueError(f"code {out_code} not in deck {deck.name!r}")
    section = list(deck.main) if in_main else list(deck.extra)
    have = section.count(out_code)
    want = have if n is None else n
    if want < 1 or want > have:
        raise ValueError(f"cannot swap {want} of {have} copies of {out_code} in {deck.name!r}")
    done = 0
    for i, c in enumerate(section):
        if c == out_code and done < want:
            section[i] = in_code
            done += 1
    main = tuple(section) if in_main else deck.main
    extra = tuple(section) if in_extra else deck.extra
    out = Deck(main, extra, deck.side, deck.name)
    if len(out.main) != len(deck.main) or len(out.extra) != len(deck.extra):
        raise ValueError("swap changed deck lengths (unreachable)")
    return out


def apply_swaps(deck: Deck, swaps: list[tuple[int, int, int | None]]) -> Deck:
    """`swap` applied in order; each item is (out_code, in_code, n_or_None)."""
    for out_code, in_code, n in swaps:
        deck = swap(deck, out_code, in_code, n)
    return deck


def shuffled_codes(main: Sequence[int], seed: int, team: int) -> list[int]:
    """A copy of `main` in the order it enters the duel for `team` under `seed`.

    The deck top is the **last** element (`DuelConfig.main_of`: "the top card
    is the last one"; pre-shuffle records agree: the opening hand used to be
    the last 5 listed cards).
    """
    out = list(main)
    shuffle_main(out, seed, team)
    return out


def opening_exposure(
    deck_main: Sequence[int], seed: int, team: int, n_cards: int = HAND_SIZE
) -> list[tuple[int, int]]:
    """The first `n_cards` cards dealt: (shuffled-list index, code) of the deck top.

    Default 5 = the opening hand. The player going first draws no card on turn
    1 (core rule), so both players' openings are these 5.
    """
    order = shuffled_codes(deck_main, seed, team)
    top = order[len(order) - n_cards :]
    return [(len(order) - n_cards + i, c) for i, c in enumerate(top)]


def seen_by_turn(deck_main: Sequence[int], seed: int, team: int, turn: int) -> list[tuple[int, int]]:
    """`opening_exposure` extended by one draw per turn: the top 5 + `turn` cards."""
    return opening_exposure(deck_main, seed, team, HAND_SIZE + turn)


def write_ydk(deck: Deck, path: str | Path) -> Path:
    """Serialize a deck (e.g. a swap variant) to a `.ydk` file."""
    path = Path(path)
    lines = ["#main", *map(str, deck.main), "#extra", *map(str, deck.extra), "!side",
             *map(str, deck.side), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
