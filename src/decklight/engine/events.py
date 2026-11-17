"""Public history events: the opponent rows a client rebuilds from public messages (NOTEBOOK 2026-11-10 12:58).

`public_events` is the one rule both sides use: the live client (`play.stream.history_from_message`, which keeps
the events of the viewer's opponent) and the engine envs (`obs_history`, train = deploy), so the observable history
a student trains on is built exactly as the deployed bot sees it.
"""

from __future__ import annotations

import struct
from collections.abc import Callable

from decklight.engine import constants as C
from decklight.engine.actions import Kind
from decklight.engine.messages import CardRef
from decklight.engine.obs import HistoryEntry, i32, loc_id


# The messages `public_events` maps (the rest yield nothing).
EVENT_MSGS = frozenset({C.MSG_SUMMONING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING, C.MSG_SET, C.MSG_CHAINING,
                        C.MSG_ATTACK, C.MSG_POS_CHANGE, C.MSG_NEW_PHASE})  # fmt: skip


def _loc_info(raw: bytes, off: int) -> tuple[int, int, int, int]:
    con, loc = raw[off], raw[off + 1]
    (seq,) = struct.unpack_from("<I", raw, off + 2)
    (pos,) = struct.unpack_from("<I", raw, off + 6)
    return con, loc, seq, pos


def public_events(raw: bytes, turn: int, phase: int, turn_player: int, chain_open: bool,
                  mzone_code: Callable[[int, int], int]) -> list[HistoryEntry]:  # fmt: skip
    """History entries of whoever acts in a public message, for any actor ([] when unmappable). The state
    arguments are the ones after the message was applied (turn / phase / chain).

    Covers summon / special / flip summon, activation (MSG_CHAINING), attack, set (code
    hidden), position change, and phase moves (TO_BATTLE / TO_MAIN2 / TO_END on
    MSG_NEW_PHASE, the turn player's). PASS / NO / pick steps are invisible to a client and
    never emitted. Rows for summons made while a chain is open are skipped too: those come
    from effect resolution, which takes no decision (the activation row already covers it).

    Summons and sets emit TWO rows: the engine records the pick (SUMMON/MSET/SSET, always
    from IDLE) and the placement pick (PLACE, always SELECT_PLACE) separately, and both
    are deterministic from the event (zone = the summoned/set card's zone). Singleton
    placements (one zone free) take no decision in the engine and are over-counted here;
    tribute/material picks are invisible and under-counted. Deciding-player message types
    are deterministic (IDLE/BATTLE/PLACE); activations (IDLE/BATTLE/CHAIN) and special
    summons keep the informing message type. `mzone_code(con, seq)` names an attacker.
    """
    t = raw[0]
    out: list[HistoryEntry] = []

    def entry(
        kind: Kind, ref: CardRef, code: int, desc: int, arg: int, actor: int, public: bool, msg_type: int
    ) -> HistoryEntry:
        row = [
            int(kind),
            -1,
            code,
            i32(desc >> 20),
            desc & 0xFFFFF,
            arg,
            msg_type,
            loc_id(ref.location),
            ref.controller ^ actor,
            0,
        ]
        return HistoryEntry(row, actor, turn, phase, public)

    def place_row(ref: CardRef, actor: int) -> HistoryEntry:
        zone = ref.sequence if ref.location == C.LOCATION_MZONE else ref.sequence + 8
        return entry(Kind.PLACE, ref, 0, 0, zone, actor, False, C.MSG_SELECT_PLACE)

    if t == C.MSG_SUMMONING and len(raw) >= 15:
        (code,) = struct.unpack_from("<I", raw, 1)
        con, loc, seq, pos = _loc_info(raw, 5)
        if not chain_open:
            ref = CardRef(code, con, loc, seq, pos)
            out += [entry(Kind.SUMMON, ref, code, 0, 0, con, True, C.MSG_SELECT_IDLECMD), place_row(ref, con)]
    elif t == C.MSG_SPSUMMONING and len(raw) >= 15:
        (code,) = struct.unpack_from("<I", raw, 1)
        con, loc, seq, pos = _loc_info(raw, 5)
        if not chain_open:
            ref = CardRef(code, con, loc, seq, pos)
            out += [
                entry(Kind.SPSUMMON, ref, code, 0, 0, con, True, C.MSG_SELECT_IDLECMD),
                place_row(ref, con),
            ]
    elif t == C.MSG_FLIPSUMMONING and len(raw) >= 15:
        (code,) = struct.unpack_from("<I", raw, 1)
        con, loc, seq, pos = _loc_info(raw, 5)
        out.append(
            entry(Kind.REPOS, CardRef(code, con, loc, seq, pos), code, 0, 0, con, True, C.MSG_SELECT_IDLECMD)
        )
    elif t == C.MSG_SET and len(raw) >= 15:
        con, loc, seq, pos = _loc_info(raw, 5)
        kind = Kind.MSET if loc == C.LOCATION_MZONE else Kind.SSET
        ref = CardRef(0, con, loc, seq, pos)
        out += [entry(kind, ref, 0, 0, 0, con, False, C.MSG_SELECT_IDLECMD), place_row(ref, con)]
    elif t == C.MSG_CHAINING and len(raw) >= 29:
        (code,) = struct.unpack_from("<I", raw, 1)
        con, loc, seq, pos = _loc_info(raw, 5)
        tcon = raw[15]
        (desc,) = struct.unpack_from("<Q", raw, 21)
        public = bool(pos & C.POS_FACEUP)
        out.append(entry(Kind.ACTIVATE, CardRef(code, con, loc, seq, pos), code, desc, 0, tcon, public, t))
    elif t == C.MSG_ATTACK and len(raw) >= 21:
        con, loc, seq, pos = _loc_info(raw, 1)
        _tcon, _tloc, _tseq, _tpos = _loc_info(raw, 11)
        code = mzone_code(con, seq)
        direct = _tloc == 0 and _tcon == 0 and _tseq == 0
        out.append(
            entry(
                Kind.ATTACK,
                CardRef(code, con, loc, seq, pos),
                code,
                0,
                int(direct),
                con,
                True,
                C.MSG_SELECT_BATTLECMD,
            )
        )
    elif t == C.MSG_POS_CHANGE and len(raw) >= 10:
        (code,) = struct.unpack_from("<I", raw, 1)
        cc, cl, cs, _pp, cp = raw[5], raw[6], raw[7], raw[8], raw[9]
        if not chain_open:
            out.append(
                entry(
                    Kind.REPOS,
                    CardRef(code, cc, cl, cs, cp),
                    code,
                    0,
                    0,
                    cc,
                    bool(cp & C.POS_FACEUP),
                    C.MSG_SELECT_IDLECMD,
                )
            )
    elif t == C.MSG_NEW_PHASE and len(raw) >= 3:
        (new_phase,) = struct.unpack_from("<H", raw, 1)
        kind_msg = {
            C.PHASE_BATTLE_START: (Kind.TO_BATTLE, C.MSG_SELECT_IDLECMD),
            C.PHASE_MAIN2: (Kind.TO_MAIN2, C.MSG_SELECT_BATTLECMD),
            C.PHASE_END: (Kind.TO_END, C.MSG_SELECT_BATTLECMD),
        }.get(new_phase)
        if kind_msg is not None:
            kind, msg_type = kind_msg
            actor = turn_player
            out.append(entry(kind, CardRef(0, actor, 0, 0), 0, 0, 0, actor, True, msg_type))
    return out
