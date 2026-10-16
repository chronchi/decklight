"""Per-player filtering of core messages and board refresh packets (1v1, no observers).

A port of generic_duel.cpp `Sending` / `BeforeParsing` / `AfterParsing` for a single
human viewer (engine seat 0 or 1). The bot gets nothing over the network.

- `filtered_message(raw, viewer)`: the STOC_GAME_MSG payload for `viewer`, or None
  when the viewer must not see this message.
- `waiting_player(raw)`: the engine seat that must answer, if `raw` awaits a response.
- `before_refreshes(raw)` / `after_refreshes(raw)`: board queries the host must run
  around the message, as ("loc", player, location[, flags]) or
  ("single", player, location, seq[, flags]) tuples executed with Duel.query_location.
- `encode_update_data` / `encode_update_card`: MSG_UPDATE_DATA / MSG_UPDATE_CARD
  packets from parsed query rows (ygo_zero.engine.query.CardState), public-filtered
  for the opponent exactly like CoreUtils::Query::GenerateBuffer(public).
"""

import struct

from ygo_zero.engine import constants as C
from ygo_zero.engine import messages as M
from ygo_zero.engine.query import CardState

LOC_INFO = 10  # con u8, loc u8, seq u32, pos u32

_FACEUP = C.POS_FACEUP
_FACEDOWN = C.POS_FACEDOWN


def _u8(b: bytes, i: int) -> int:
    return b[i]


def _u32(b: bytes, i: int) -> int:
    return struct.unpack_from("<I", b, i)[0]


def _loc(b: bytes, i: int) -> tuple[int, int, int, int]:
    """loc_info at i -> (controller, location, sequence, position)."""
    con, loc = b[i], b[i + 1]
    (seq,) = struct.unpack_from("<I", b, i + 2)
    (pos,) = struct.unpack_from("<I", b, i + 6)
    return con, loc, seq, pos


def filtered_message(raw: bytes, viewer: int) -> bytes | None:
    """The copy of core message `raw` for engine seat `viewer` (None = send nothing)."""
    t = raw[0]
    if t == C.MSG_HINT:
        htype, player = raw[1], raw[2]
        if htype in (1, 2, 3, 5) or htype == 200:
            return raw if viewer == player else None
        if htype in (4, 6, 7, 8, 9, 11):
            return raw if viewer != player else None
        if htype in (10, 201, 202, 203):
            return raw
        return None
    if t in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD, C.MSG_WIN):
        if t == C.MSG_WIN:
            return raw
        return raw if viewer == raw[1] else None
    if t in (
        C.MSG_SELECT_EFFECTYN, C.MSG_SELECT_YESNO, C.MSG_SELECT_OPTION, C.MSG_SELECT_CHAIN,
        C.MSG_SELECT_PLACE, C.MSG_SELECT_DISFIELD, C.MSG_SELECT_POSITION, C.MSG_SELECT_COUNTER,
        C.MSG_SELECT_SUM, C.MSG_SORT_CARD, C.MSG_SORT_CHAIN, C.MSG_ROCK_PAPER_SCISSORS,
        C.MSG_ANNOUNCE_RACE, C.MSG_ANNOUNCE_ATTRIB, C.MSG_ANNOUNCE_CARD, C.MSG_ANNOUNCE_NUMBER,
    ):
        return raw if viewer == raw[1] else None
    if t == C.MSG_SELECT_CARD:
        player = raw[1]
        if viewer != player:
            return None
        (count,) = struct.unpack_from("<I", raw, 11)
        out = bytearray(raw)
        pos = 15
        for _ in range(count):
            con = out[pos + 4]
            if con != player:
                struct.pack_into("<I", out, pos, 0)
            pos += 14
        return bytes(out)
    if t == C.MSG_SELECT_TRIBUTE:
        player = raw[1]
        if viewer != player:
            return None
        (count,) = struct.unpack_from("<I", raw, 11)
        out = bytearray(raw)
        pos = 15
        for _ in range(count):
            if out[pos + 4] != player:  # controller u8 right after the code
                struct.pack_into("<I", out, pos, 0)
            pos += 11
        return bytes(out)
    if t == C.MSG_SELECT_UNSELECT_CARD:
        player = raw[1]
        if viewer != player:
            return None
        out = bytearray(raw)
        pos = 12
        for _ in range(2):
            (count,) = struct.unpack_from("<I", out, pos)
            pos += 4
            for _ in range(count):
                if out[pos + 4] != player:
                    struct.pack_into("<I", out, pos, 0)
                pos += 14
        return bytes(out)
    if t == C.MSG_CONFIRM_CARDS:
        player = raw[1]
        (count,) = struct.unpack_from("<I", raw, 2)
        if count == 0:
            return raw
        loc = raw[11]  # first card's location (code u32, controller u8, location u8)
        if loc in (C.LOCATION_DECK, C.LOCATION_EXTRA):
            return raw if viewer == player else None
        return raw
    if t in (C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_EXTRA):
        player = raw[1]
        if viewer == player:
            return raw
        (count,) = struct.unpack_from("<I", raw, 2)
        out = bytearray(raw)
        struct.pack_into(f"<{count}I", out, 6, *([0] * count))
        return bytes(out)
    if t == C.MSG_MOVE:
        _code = _u32(raw, 1)
        _prev = _loc(raw, 5)
        con, loc, _seq, pos = _loc(raw, 15)
        if viewer == con:
            return raw
        if not (loc & (C.LOCATION_GRAVE + C.LOCATION_OVERLAY)) and (
            (loc & (C.LOCATION_DECK + C.LOCATION_HAND)) or (pos & _FACEDOWN)
        ):
            out = bytearray(raw)
            struct.pack_into("<I", out, 1, 0)
            return bytes(out)
        return raw
    if t == C.MSG_SET:
        out = bytearray(raw)
        struct.pack_into("<I", out, 1, 0)
        return bytes(out)
    if t == C.MSG_SPSUMMONING:
        con, _, _seq, pos = _loc(raw, 5)
        if pos & _FACEDOWN == 0:
            return raw
        if viewer == con:
            return raw
        out = bytearray(raw)
        struct.pack_into("<I", out, 1, 0)
        return bytes(out)
    if t == C.MSG_DRAW:
        player = raw[1]
        if viewer == player:
            return raw
        (count,) = struct.unpack_from("<I", raw, 2)
        out = bytearray(raw)
        pos = 6
        for _ in range(count):
            (p,) = struct.unpack_from("<I", out, pos + 4)
            if not p & _FACEUP:
                struct.pack_into("<I", out, pos, 0)
            pos += 8
        return bytes(out)
    return raw


def waiting_player(raw: bytes) -> int | None:
    """The engine seat that must answer `raw`, if it awaits a response (else None)."""
    if raw[0] in M.SELECT_TYPES:
        return raw[1]
    return None


def _both(loc: int) -> list[tuple]:
    return [("loc", 0, loc), ("loc", 1, loc)]


def before_refreshes(raw: bytes) -> list[tuple]:
    """Queries EDOPro runs before sending the message (BeforeParsing)."""
    t = raw[0]
    if t in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD):
        return _both(C.LOCATION_MZONE) + _both(C.LOCATION_SZONE) + _both(C.LOCATION_HAND)
    if t in (C.MSG_SELECT_CHAIN, C.MSG_NEW_TURN):
        return _both(C.LOCATION_MZONE) + _both(C.LOCATION_SZONE)
    if t == C.MSG_FLIPSUMMONING:
        con, loc, seq, _pos = _loc(raw, 5)
        return [("single", con, loc, seq)]
    return []


def after_refreshes(raw: bytes) -> list[tuple]:
    """Queries EDOPro runs after sending the message (AfterParsing)."""
    t = raw[0]
    if t in (C.MSG_SHUFFLE_HAND, C.MSG_DRAW):
        return [("loc", raw[1], C.LOCATION_HAND)]
    if t == C.MSG_SHUFFLE_EXTRA:
        return [("loc", raw[1], C.LOCATION_EXTRA)]
    if t == C.MSG_SWAP_GRAVE_DECK:
        return [("loc", raw[1], C.LOCATION_GRAVE)]
    if t == C.MSG_REVERSE_DECK:
        return []
    if t == C.MSG_SHUFFLE_SET_CARD:
        return [("loc", 0, raw[2]), ("loc", 1, raw[2])]
    if t == C.MSG_NEW_TURN:
        return []
    if t in (
        C.MSG_DAMAGE_STEP_START, C.MSG_DAMAGE_STEP_END, C.MSG_SUMMONED, C.MSG_SPSUMMONED,
        C.MSG_FLIPSUMMONED, C.MSG_NEW_PHASE, C.MSG_CHAINED, C.MSG_CHAIN_SOLVED, C.MSG_CHAIN_END,
    ):
        out = _both(C.LOCATION_MZONE)
        if t not in (C.MSG_DAMAGE_STEP_START, C.MSG_DAMAGE_STEP_END):
            out += _both(C.LOCATION_SZONE)
        if t in (C.MSG_NEW_PHASE, C.MSG_CHAINED, C.MSG_CHAIN_END):
            out += _both(C.LOCATION_HAND)
        return out
    if t == C.MSG_MOVE:
        _pcon, ploc, _pseq, _ppos = _loc(raw, 5)
        ccon, cloc, cseq, _cpos = _loc(raw, 15)
        if cloc != 0 and not cloc & 0x80 and (cloc != ploc or _pcon != ccon):
            return [("single", ccon, cloc, cseq)]
        return []
    if t == C.MSG_POS_CHANGE:
        cc, cl, cs, pp, cp = raw[5], raw[6], raw[7], raw[8], raw[9]
        if (pp & _FACEDOWN) and (cp & _FACEUP):
            return [("single", cc, cl, cs)]
        return []
    if t == C.MSG_SWAP:
        pcon, ploc, pseq, _p = _loc(raw, 5)
        ccon, cloc, cseq, _c = _loc(raw, 19)
        return [("single", pcon, ploc, pseq), ("single", ccon, cloc, cseq)]
    return []


# --- MSG_UPDATE_DATA / MSG_UPDATE_CARD encoders (CoreUtils::Query::GenerateBuffer) ---

# (flag, kind): kinds mirror core_utils.cpp GetFlagSize; ascending flag order.
_FIELDS: tuple[tuple[int, str], ...] = (
    (C.QUERY_CODE, "u32"), (C.QUERY_POSITION, "u32"), (C.QUERY_TYPE, "u32"),
    (C.QUERY_LEVEL, "u32"), (C.QUERY_RANK, "u32"), (C.QUERY_ATTRIBUTE, "u32"),
    (C.QUERY_RACE, "u64"), (C.QUERY_ATTACK, "i32"), (C.QUERY_DEFENSE, "i32"),
    (C.QUERY_BASE_ATTACK, "i32"), (C.QUERY_BASE_DEFENSE, "i32"),
    (C.QUERY_OVERLAY_CARD, "codes"), (C.QUERY_OWNER, "u8"), (C.QUERY_STATUS, "u32"),
    (C.QUERY_IS_PUBLIC, "u8"), (C.QUERY_LSCALE, "u32"), (C.QUERY_RSCALE, "u32"),
    (C.QUERY_LINK, "link"),
)
# Flags stripped for the opponent unless the card is face-up/public (IsPublicQuery).
_PRIVATE = frozenset({
    C.QUERY_CODE, C.QUERY_TYPE, C.QUERY_LEVEL, C.QUERY_RANK, C.QUERY_ATTRIBUTE,
    C.QUERY_RACE, C.QUERY_ATTACK, C.QUERY_DEFENSE, C.QUERY_BASE_ATTACK,
    C.QUERY_BASE_DEFENSE, C.QUERY_STATUS, C.QUERY_LSCALE, C.QUERY_RSCALE, C.QUERY_LINK,
})


def _field_value(flag: int, kind: str, card: CardState) -> bytes:
    if kind == "u8":
        v = {C.QUERY_OWNER: card.owner, C.QUERY_IS_PUBLIC: int(card.public)}[flag]
        return struct.pack("<B", v)
    if kind == "u64":
        return struct.pack("<Q", card.race)
    if kind == "i32":
        v = {
            C.QUERY_ATTACK: card.attack, C.QUERY_DEFENSE: card.defense,
            C.QUERY_BASE_ATTACK: card.base_attack, C.QUERY_BASE_DEFENSE: card.base_defense,
        }[flag]
        return struct.pack("<i", v)
    if kind == "codes":
        return struct.pack(f"<{len(card.overlay) + 1}I", len(card.overlay), *card.overlay)
    if kind == "link":
        return struct.pack("<II", card.link, card.link_marker)
    v = {
        C.QUERY_CODE: card.code, C.QUERY_POSITION: card.position, C.QUERY_TYPE: card.type,
        C.QUERY_LEVEL: card.level, C.QUERY_RANK: card.rank,
        C.QUERY_ATTRIBUTE: card.attribute, C.QUERY_STATUS: card.status,
        C.QUERY_LSCALE: card.lscale, C.QUERY_RSCALE: card.rscale,
    }[flag]
    return struct.pack("<I", v & 0xFFFFFFFF)


def encode_record(card: CardState | None, public_only: bool) -> bytes:
    """One QueryLocation slot (empty slot = u16 0; record ends with QUERY_END)."""
    if card is None:
        return b"\x00\x00"
    public = bool(card.position & _FACEUP) or card.public
    out = bytearray()
    for flag, kind in _FIELDS:
        if public_only and not public and flag in _PRIVATE:
            continue
        val = _field_value(flag, kind, card)
        out += struct.pack("<H", len(val) + 4) + struct.pack("<I", flag) + val
    out += struct.pack("<HI", 4, C.QUERY_END)
    return bytes(out)


def encode_update_data(player: int, location: int, cards: list[CardState | None],
                       public_only: bool) -> bytes:
    recs = b"".join(encode_record(c, public_only) for c in cards)
    return bytes([C.MSG_UPDATE_DATA, player, location]) + struct.pack("<I", len(recs)) + recs


def encode_update_card(player: int, location: int, seq: int, card: CardState | None,
                       public_only: bool) -> bytes:
    return bytes([C.MSG_UPDATE_CARD, player, location, seq & 0xFF]) + encode_record(
        card, public_only
    )
