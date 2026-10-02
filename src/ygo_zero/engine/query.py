"""Read the board through `OCG_DuelQueryLocation` / `OCG_DuelQueryField` (formats from `card::get_infos`
and `OCG_DuelQueryField` in ocgapi.cpp).

The core reports everything it knows, hidden cards included; hiding is the observer's job (`obs.py`).
"""

import struct
from dataclasses import dataclass

from ygo_zero.engine import constants as C

CARD_FLAGS = (
    C.QUERY_CODE | C.QUERY_POSITION | C.QUERY_TYPE | C.QUERY_LEVEL | C.QUERY_RANK | C.QUERY_ATTRIBUTE
    | C.QUERY_RACE | C.QUERY_ATTACK | C.QUERY_DEFENSE | C.QUERY_BASE_ATTACK | C.QUERY_BASE_DEFENSE
    | C.QUERY_OVERLAY_CARD | C.QUERY_OWNER | C.QUERY_STATUS | C.QUERY_IS_PUBLIC | C.QUERY_LSCALE
    | C.QUERY_RSCALE | C.QUERY_LINK
)  # fmt: skip

# Locations queried, in the order the core lists them per player.
QUERY_LOCATIONS = (
    C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE,
    C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA,
)  # fmt: skip


@dataclass(slots=True)
class CardState:
    controller: int
    location: int
    sequence: int  # zone / list index; for an overlay unit: the holder's zone
    code: int = 0
    position: int = 0
    type: int = 0
    level: int = 0
    rank: int = 0
    attribute: int = 0
    race: int = 0
    attack: int = 0
    defense: int = 0
    base_attack: int = 0
    base_defense: int = 0
    owner: int = 0
    status: int = 0
    public: bool = False
    lscale: int = 0
    rscale: int = 0
    link: int = 0
    link_marker: int = 0
    overlay: tuple[int, ...] = ()  # codes of the xyz materials attached to this card
    overlay_index: int = 0  # for an overlay unit: index in the holder's material list


def _i32(v: int) -> int:
    return v - (1 << 32) if v >= 1 << 31 else v


def parse_card(buf: bytes, pos: int, card: CardState) -> int:
    """Fill `card` from one get_infos record starting at `pos`; returns the position after QUERY_END."""
    while True:
        size, flag = struct.unpack_from("<HI", buf, pos)
        body = pos + 6
        pos += 2 + size
        if flag == C.QUERY_END:
            return pos
        if flag == C.QUERY_RACE:
            card.race = struct.unpack_from("<Q", buf, body)[0]
        elif flag in (C.QUERY_OWNER, C.QUERY_IS_PUBLIC, C.QUERY_IS_HIDDEN):
            v = buf[body]
            if flag == C.QUERY_OWNER:
                card.owner = v
            elif flag == C.QUERY_IS_PUBLIC:
                card.public = bool(v)
        elif flag == C.QUERY_LINK:
            card.link, card.link_marker = struct.unpack_from("<II", buf, body)
        elif flag == C.QUERY_OVERLAY_CARD:
            (n,) = struct.unpack_from("<I", buf, body)
            card.overlay = struct.unpack_from(f"<{n}I", buf, body + 4)
        elif size == 8:
            v = struct.unpack_from("<I", buf, body)[0]
            match flag:
                case C.QUERY_CODE:
                    card.code = v
                case C.QUERY_POSITION:
                    card.position = v
                case C.QUERY_TYPE:
                    card.type = v
                case C.QUERY_LEVEL:
                    card.level = v
                case C.QUERY_RANK:
                    card.rank = v
                case C.QUERY_ATTRIBUTE:
                    card.attribute = v
                case C.QUERY_ATTACK:
                    card.attack = _i32(v)
                case C.QUERY_DEFENSE:
                    card.defense = _i32(v)
                case C.QUERY_BASE_ATTACK:
                    card.base_attack = _i32(v)
                case C.QUERY_BASE_DEFENSE:
                    card.base_defense = _i32(v)
                case C.QUERY_STATUS:
                    card.status = v
                case C.QUERY_LSCALE:
                    card.lscale = v
                case C.QUERY_RSCALE:
                    card.rscale = v


def parse_location(buf: bytes, controller: int, location: int) -> list[CardState | None]:
    """A QueryLocation buffer: u32 total size, then per slot either a record or an empty i16 0."""
    out: list[CardState | None] = []
    pos, end = 4, len(buf)
    seq = 0
    while pos < end:
        (size,) = struct.unpack_from("<H", buf, pos)
        if size == 0:
            out.append(None)
            pos += 2
        else:
            card = CardState(controller, location, seq)
            pos = parse_card(buf, pos, card)
            out.append(card)
        seq += 1
    return out


@dataclass(slots=True)
class ChainLink:
    code: int
    controller: int  # handler's current controller
    location: int
    sequence: int
    position: int
    triggering_controller: int
    triggering_location: int
    triggering_sequence: int
    description: int


def parse_chain(buf: bytes) -> list[ChainLink]:
    """The chain section at the end of a QueryField buffer."""
    pos = 4  # duel options
    for _ in range(2):
        pos += 4  # lp
        for _ in range(7 + 8):  # mzone + szone slots
            occupied = buf[pos]
            pos += 1 + (5 if occupied else 0)
        pos += 6 * 4
    (n,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    chain = []
    for _ in range(n):
        code, con, loc, seq, position, tcon, tloc, tseq, desc = struct.unpack_from("<IBBIIBBIQ", buf, pos)
        pos += 4 + 1 + 1 + 4 + 4 + 1 + 1 + 4 + 8
        chain.append(ChainLink(code, con, loc, seq, position, tcon, tloc, tseq, desc))
    return chain


def lps(buf: bytes) -> tuple[int, int]:
    """Both players' LP from a QueryField buffer."""
    lp0 = struct.unpack_from("<I", buf, 4)[0]
    pos = 8
    for _ in range(7 + 8):
        pos += 1 + (5 if buf[pos] else 0)
    pos += 6 * 4
    return lp0, struct.unpack_from("<I", buf, pos)[0]
