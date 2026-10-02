"""Encode responses for OCG_DuelSetResponse. Formats follow the `returns.at<T>(i)` reads in `playerop.cpp`."""

import struct
from collections.abc import Iterable, Sequence

# SELECT_IDLECMD action types: response = (index << 16) | type
IDLE_SUMMON, IDLE_SPSUMMON, IDLE_REPOS, IDLE_MSET, IDLE_SSET, IDLE_ACTIVATE = 0, 1, 2, 3, 4, 5
IDLE_BATTLE, IDLE_END, IDLE_SHUFFLE = 6, 7, 8
# SELECT_BATTLECMD action types
BATTLE_ACTIVATE, BATTLE_ATTACK, BATTLE_MAIN2, BATTLE_END = 0, 1, 2, 3


def int32(value: int) -> bytes:
    """YESNO / EFFECTYN (0|1), OPTION, ANNOUNCE_NUMBER (index), CHAIN (index or -1),
    POSITION (single POS_* bit), ANNOUNCE_CARD (code), ROCK_PAPER_SCISSORS (1..3)."""
    return struct.pack("<i", value)


def command(action: int, index: int = 0) -> bytes:
    """SELECT_IDLECMD / SELECT_BATTLECMD."""
    return struct.pack("<i", (index << 16) | action)


def cards(indices: Sequence[int]) -> bytes:
    """SELECT_CARD / SELECT_TRIBUTE / SELECT_SUM: indices into the message's card list (type 2 = u8 indices)."""
    return struct.pack("<iI", 2, len(indices)) + bytes(indices)


# SELECT_CARD/TRIBUTE (if cancelable), UNSELECT_CARD (cancel/finish), CHAIN (pass)
CANCEL = struct.pack("<i", -1)


def unselect_card(index: int) -> bytes:
    """SELECT_UNSELECT_CARD: one index into selectable + unselectable (concatenated)."""
    return struct.pack("<ii", 1, index)


def place(zones: Iterable[tuple[int, int, int]]) -> bytes:
    """SELECT_PLACE / SELECT_DISFIELD: (absolute player, LOCATION_MZONE|SZONE, sequence) per zone."""
    return b"".join(bytes(z) for z in zones)


def counters(counts: Sequence[int]) -> bytes:
    """SELECT_COUNTER: counters to remove from each listed card."""
    return struct.pack(f"<{len(counts)}h", *counts)


def sort(order: Sequence[int] | None) -> bytes:
    """SORT_CARD / SORT_CHAIN: a permutation, or None to keep the default order."""
    return b"\xff" if order is None else bytes(order)


def announce_race(races: int) -> bytes:
    return struct.pack("<Q", races)


def announce_attrib(attributes: int) -> bytes:
    return struct.pack("<I", attributes)
