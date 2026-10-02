"""Parse the core's message buffer.

`OCG_DuelGetMessage` returns `[u32 size][size bytes]...`; each payload is one message whose first
byte is the MSG_* type. Every message that awaits a response (the SELECT_*/ANNOUNCE_*/SORT_* family,
written in `playerop.cpp`) is parsed in full; of the rest, only a few that drive the loop
(WIN, NEW_TURN, NEW_PHASE, HINT, RETRY) are decoded. Everything else stays `Message(type, raw)`.

Layout notes (see ../ygo-architect/CLAUDE.md for the history):
- `loc_info` = controller u8, location u8, sequence u32, position u32 (10 bytes).
- Several lists use a short `con u8, loc u8, seq u8` form instead (repos, attackable, counter).
"""

import struct
from dataclasses import dataclass, field
from typing import NamedTuple

from ygo_zero.engine import constants as C


class CardRef(NamedTuple):
    code: int
    controller: int
    location: int
    sequence: int
    position: int = 0


class ChainOption(NamedTuple):
    card: CardRef
    description: int
    client_mode: int


@dataclass(slots=True)
class Message:
    type: int
    raw: bytes = field(repr=False)


@dataclass(slots=True)
class Win(Message):
    player: int  # 2 = draw
    reason: int


@dataclass(slots=True)
class NewTurn(Message):
    player: int


@dataclass(slots=True)
class NewPhase(Message):
    phase: int


@dataclass(slots=True)
class Hint(Message):
    hint_type: int
    player: int
    data: int


# --- messages awaiting a response; all have `player` ---


@dataclass(slots=True)
class Select(Message):
    player: int


@dataclass(slots=True)
class SelectIdleCmd(Select):
    summonable: list[CardRef]
    spsummonable: list[CardRef]
    repositionable: list[CardRef]
    msetable: list[CardRef]
    ssetable: list[CardRef]
    activatable: list[ChainOption]
    can_battle_phase: bool
    can_end_phase: bool
    can_shuffle: bool


@dataclass(slots=True)
class SelectBattleCmd(Select):
    activatable: list[ChainOption]
    attackable: list[tuple[CardRef, bool]]  # (card, can attack directly)
    can_main2: bool
    can_end_phase: bool


@dataclass(slots=True)
class SelectEffectYn(Select):
    card: CardRef
    description: int


@dataclass(slots=True)
class SelectYesNo(Select):
    description: int


@dataclass(slots=True)
class SelectOption(Select):
    options: list[int]


@dataclass(slots=True)
class SelectCard(Select):
    cancelable: bool
    min: int
    max: int
    cards: list[CardRef]


@dataclass(slots=True)
class SelectUnselectCard(Select):
    finishable: bool
    cancelable: bool
    min: int
    max: int
    selectable: list[CardRef]
    unselectable: list[CardRef]


@dataclass(slots=True)
class SelectChain(Select):
    spe_count: int
    forced: bool
    hint_timing: int
    hint_timing_other: int
    chains: list[ChainOption]


@dataclass(slots=True)
class SelectPlace(Select):  # also MSG_SELECT_DISFIELD
    count: int
    flag: int  # set bits are UNavailable; bits 0-15 = the selecting player's zones


@dataclass(slots=True)
class SelectPosition(Select):
    code: int
    positions: int


@dataclass(slots=True)
class SelectTribute(Select):
    cancelable: bool
    min: int
    max: int
    cards: list[tuple[CardRef, int]]  # (card, release_param)


@dataclass(slots=True)
class SelectCounter(Select):
    counter_type: int
    count: int
    cards: list[tuple[CardRef, int]]  # (card, counters on it)


@dataclass(slots=True)
class SelectSum(Select):
    at_least: bool  # True: sum >= acc with no redundant card; False: exact sum, min..max cards
    acc: int
    min: int
    max: int
    must: list[tuple[CardRef, int]]  # (card, sum_param: low16 | alt << 16)
    cards: list[tuple[CardRef, int]]


@dataclass(slots=True)
class SortCards(Select):  # MSG_SORT_CARD and MSG_SORT_CHAIN
    cards: list[CardRef]


@dataclass(slots=True)
class AnnounceRace(Select):
    count: int
    available: int


@dataclass(slots=True)
class AnnounceAttrib(Select):
    count: int
    available: int


@dataclass(slots=True)
class AnnounceCard(Select):
    opcodes: list[int]


@dataclass(slots=True)
class AnnounceNumber(Select):
    options: list[int]


@dataclass(slots=True)
class RockPaperScissors(Select):
    pass


class _Reader:
    __slots__ = ("b", "p")

    def __init__(self, b: bytes) -> None:
        self.b, self.p = b, 1  # skip the type byte

    def _r(self, fmt: str, n: int) -> int:
        v = struct.unpack_from(fmt, self.b, self.p)[0]
        self.p += n
        return v

    def u8(self) -> int:
        return self._r("<B", 1)

    def u16(self) -> int:
        return self._r("<H", 2)

    def u32(self) -> int:
        return self._r("<I", 4)

    def u64(self) -> int:
        return self._r("<Q", 8)

    def card_info(self) -> CardRef:  # code + loc_info
        return CardRef(self.u32(), self.u8(), self.u8(), self.u32(), self.u32())

    def card_seq32(self) -> CardRef:  # code, con u8, loc u8, seq u32
        return CardRef(self.u32(), self.u8(), self.u8(), self.u32())

    def card_seq8(self) -> CardRef:  # code, con u8, loc u8, seq u8
        return CardRef(self.u32(), self.u8(), self.u8(), self.u8())

    def chain_short(self) -> ChainOption:  # idle/battle activatable
        return ChainOption(self.card_seq32(), self.u64(), self.u8())

    def chain_full(self) -> ChainOption:  # SELECT_CHAIN
        return ChainOption(self.card_info(), self.u64(), self.u8())

    def list(self, fn):  # u32 count prefix
        return [fn() for _ in range(self.u32())]


def _parse(t: int, raw: bytes) -> Message:
    r = _Reader(raw)
    match t:
        case C.MSG_WIN:
            return Win(t, raw, r.u8(), r.u8())
        case C.MSG_NEW_TURN:
            return NewTurn(t, raw, r.u8())
        case C.MSG_NEW_PHASE:
            return NewPhase(t, raw, r.u16())
        case C.MSG_HINT:
            return Hint(t, raw, r.u8(), r.u8(), r.u64())
        case C.MSG_SELECT_IDLECMD:
            p = r.u8()
            lists = [r.list(r.card_seq32), r.list(r.card_seq32), r.list(r.card_seq8)]
            lists += [r.list(r.card_seq32), r.list(r.card_seq32), r.list(r.chain_short)]
            return SelectIdleCmd(t, raw, p, *lists, bool(r.u8()), bool(r.u8()), bool(r.u8()))
        case C.MSG_SELECT_BATTLECMD:
            p = r.u8()
            acts = r.list(r.chain_short)
            atks = r.list(lambda: (r.card_seq8(), bool(r.u8())))
            return SelectBattleCmd(t, raw, p, acts, atks, bool(r.u8()), bool(r.u8()))
        case C.MSG_SELECT_EFFECTYN:
            return SelectEffectYn(t, raw, r.u8(), r.card_info(), r.u64())
        case C.MSG_SELECT_YESNO:
            return SelectYesNo(t, raw, r.u8(), r.u64())
        case C.MSG_SELECT_OPTION:
            p = r.u8()
            return SelectOption(t, raw, p, [r.u64() for _ in range(r.u8())])
        case C.MSG_SELECT_CARD:
            p, cancel, mn, mx = r.u8(), bool(r.u8()), r.u32(), r.u32()
            return SelectCard(t, raw, p, cancel, mn, mx, r.list(r.card_info))
        case C.MSG_SELECT_UNSELECT_CARD:
            p, fin, cancel, mn, mx = r.u8(), bool(r.u8()), bool(r.u8()), r.u32(), r.u32()
            sel = r.list(r.card_info)
            return SelectUnselectCard(t, raw, p, fin, cancel, mn, mx, sel, r.list(r.card_info))
        case C.MSG_SELECT_CHAIN:
            p, spe, forced, h1, h2 = r.u8(), r.u8(), bool(r.u8()), r.u32(), r.u32()
            return SelectChain(t, raw, p, spe, forced, h1, h2, r.list(r.chain_full))
        case C.MSG_SELECT_PLACE | C.MSG_SELECT_DISFIELD:
            return SelectPlace(t, raw, r.u8(), r.u8(), r.u32())
        case C.MSG_SELECT_POSITION:
            return SelectPosition(t, raw, r.u8(), r.u32(), r.u8())
        case C.MSG_SELECT_TRIBUTE:
            p, cancel, mn, mx = r.u8(), bool(r.u8()), r.u32(), r.u32()
            return SelectTribute(t, raw, p, cancel, mn, mx, r.list(lambda: (r.card_seq32(), r.u8())))
        case C.MSG_SELECT_COUNTER:
            p, ctype, count = r.u8(), r.u16(), r.u16()
            return SelectCounter(t, raw, p, ctype, count, r.list(lambda: (r.card_seq8(), r.u16())))
        case C.MSG_SELECT_SUM:
            p, at_least, acc, mn, mx = r.u8(), bool(r.u8()), r.u32(), r.u32(), r.u32()
            must = r.list(lambda: (r.card_info(), r.u32()))
            return SelectSum(t, raw, p, at_least, acc, mn, mx, must, r.list(lambda: (r.card_info(), r.u32())))
        case C.MSG_SORT_CARD | C.MSG_SORT_CHAIN:
            p = r.u8()
            return SortCards(t, raw, p, r.list(lambda: CardRef(r.u32(), r.u8(), r.u32(), r.u32())))
        case C.MSG_ANNOUNCE_RACE:
            return AnnounceRace(t, raw, r.u8(), r.u8(), r.u64())
        case C.MSG_ANNOUNCE_ATTRIB:
            return AnnounceAttrib(t, raw, r.u8(), r.u8(), r.u32())
        case C.MSG_ANNOUNCE_CARD:
            p = r.u8()
            return AnnounceCard(t, raw, p, [r.u64() for _ in range(r.u8())])
        case C.MSG_ANNOUNCE_NUMBER:
            p = r.u8()
            return AnnounceNumber(t, raw, p, [r.u64() for _ in range(r.u8())])
        case C.MSG_ROCK_PAPER_SCISSORS:
            return RockPaperScissors(t, raw, r.u8())
    return Message(t, raw)


SELECT_TYPES = frozenset({
    C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_IDLECMD, C.MSG_SELECT_EFFECTYN, C.MSG_SELECT_YESNO,
    C.MSG_SELECT_OPTION, C.MSG_SELECT_CARD, C.MSG_SELECT_CHAIN, C.MSG_SELECT_PLACE,
    C.MSG_SELECT_POSITION, C.MSG_SELECT_TRIBUTE, C.MSG_SORT_CHAIN, C.MSG_SELECT_COUNTER,
    C.MSG_SELECT_SUM, C.MSG_SELECT_DISFIELD, C.MSG_SORT_CARD, C.MSG_SELECT_UNSELECT_CARD,
    C.MSG_ROCK_PAPER_SCISSORS, C.MSG_ANNOUNCE_RACE, C.MSG_ANNOUNCE_ATTRIB, C.MSG_ANNOUNCE_CARD,
    C.MSG_ANNOUNCE_NUMBER,
})  # fmt: skip


def parse_one(raw: bytes) -> Message:
    return _parse(raw[0], raw)


def split_buffer(buf: bytes) -> list[bytes]:
    """Split an OCG_DuelGetMessage buffer into raw messages."""
    out, pos = [], 0
    while pos + 4 <= len(buf):
        (size,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        if pos + size > len(buf):
            raise ValueError(f"truncated message buffer at {pos}: need {size}, have {len(buf) - pos}")
        if size:
            out.append(buf[pos : pos + size])
        pos += size
    return out
