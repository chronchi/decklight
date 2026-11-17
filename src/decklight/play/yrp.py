"""EDOPro/Multirole replays (`.yrpX`, sent to every client as STOC_NEW_REPLAY) → match record v1 (ADR 0010).

Format (Multirole `src/Multirole/YGOPro/Replay.cpp`, f258e00):
- ExtendedReplayHeader (72 bytes): type 'yrpX', version, flags, timestamp, size (uncompressed body), hash,
  props[8] (LZMA), header version u64, seed[4] u64 (zero in the yrpX header; the seed is in the embedded yrp).
- Body, LZMA-compressed when flags & COMPRESSED: per team a u32 count + 40-byte UTF-16 names; duel flags u64; then
  messages `type u8, length u32, data`. Core messages pass `Replay::RecordMsg`, which drops every select/announce
  message and the player-specific hints (`recorded`); server messages MSG_START / MSG_UPDATE_DATA / MSG_UPDATE_CARD
  are added, and last one message of type 231 holding the old-format `.yrp`: header (seed[4]), names, start LP,
  start hand, draw per turn, duel flags, per duelist main (in the order added to the core, every card at sequence 0)
  + extra, extra-rule cards, then responses `length u8, data`.

The replay does not say in which order the two teams' cards were added (Multirole adds room team 0 first, which is
core team 1 when room team 1 goes first), and the order changes the duel. Nor does it say how: EDOPro servers add
every main-deck card at sequence 0, our hosts add with the list index (DuelConfig.edopro_add) and write the deck as
`edopro_main` of theirs. `to_record` therefore replays each way and order in our engine and keeps the one whose
recorded messages equal the replay's; that also proves our card scripts behave like the server's. A replay that
matches none is rejected (NOTEBOOK 2026-11-01 "A2b-2").
"""

from __future__ import annotations

import hashlib
import lzma
import struct
from dataclasses import dataclass, field

from decklight.engine import constants as C
from decklight.engine.duel import Duel, DuelConfig, Engine, canonical_trace

from . import match_record as MR

YRP1, YRPX = 0x31707279, 0x58707279
F_COMPRESSED, F_TAG, F_SINGLE, F_EXTENDED = 0x1, 0x2, 0x8, 0x200
OLD_REPLAY_FORMAT = 231
SERVER_MSGS = frozenset({C.MSG_START, C.MSG_UPDATE_DATA, C.MSG_UPDATE_CARD, OLD_REPLAY_FORMAT})
_HDR = struct.Struct("<6I8sQ4Q")  # 72 bytes
# Multirole Replay::RecordMsg: no select/announce messages, no player-specific hints (types 1, 2, 3, 5).
_UNRECORDED = frozenset({
    C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_IDLECMD, C.MSG_SELECT_EFFECTYN, C.MSG_SELECT_YESNO, C.MSG_SELECT_OPTION,
    C.MSG_SELECT_CHAIN, C.MSG_SELECT_PLACE, C.MSG_SELECT_DISFIELD, C.MSG_SELECT_POSITION, C.MSG_SELECT_COUNTER,
    C.MSG_SELECT_SUM, C.MSG_SORT_CARD, C.MSG_SORT_CHAIN, C.MSG_ROCK_PAPER_SCISSORS, C.MSG_ANNOUNCE_RACE,
    C.MSG_ANNOUNCE_ATTRIB, C.MSG_ANNOUNCE_CARD, C.MSG_ANNOUNCE_NUMBER, C.MSG_SELECT_CARD, C.MSG_SELECT_TRIBUTE,
    C.MSG_SELECT_UNSELECT_CARD,
})
_UNRECORDED_HINTS = frozenset({1, 2, 3, 5})


def recorded(msgs: list[bytes]) -> list[bytes]:
    """The messages Multirole's `Replay::RecordMsg` keeps (idempotent, so it also applies to filtered replays)."""
    return [
        m for m in msgs
        if m[0] not in _UNRECORDED and not (m[0] == C.MSG_HINT and len(m) > 1 and m[1] in _UNRECORDED_HINTS)
    ]


def edopro_main(main: list[int]) -> list[int]:
    """Our add order <-> EDOPro's (every card at sequence 0), giving the same deck: our index 1 goes to the bottom
    (core field::add_card: sequence 1 = deck bottom). Its own inverse."""
    m = list(main)
    return [m[1], m[0], *m[2:]] if len(m) > 1 else m


def yrp_decks(cfg: DuelConfig) -> list[list[tuple[list[int], list[int]]]]:
    """`Yrp.decks` for a duel our engine ran: each main deck in the order EDOPro adds it to rebuild the same deck."""
    out = []
    for team, deck in ((0, cfg.deck0), (1, cfg.deck1)):
        main = cfg.main_of(team)
        out.append([(main if cfg.edopro_add else edopro_main(main), list(deck.extra))])
    return out


@dataclass
class Yrp:
    seed: tuple[int, int, int, int]
    names: list[list[str]]
    lp: int
    hand: int
    draw: int
    flags: int
    decks: list[list[tuple[list[int], list[int]]]]  # [team][duelist] -> (main in core order, extra)
    extra_cards: list[int]
    responses: list[bytes]


@dataclass
class YrpX:
    timestamp: int
    names: list[list[str]]
    flags: int
    messages: list[bytes] = field(default_factory=list)  # type byte + data, in order
    yrp: Yrp | None = None


class _Reader:
    def __init__(self, b: bytes) -> None:
        self.b, self.p = b, 0

    def take(self, n: int) -> bytes:
        if self.p + n > len(self.b):
            raise ValueError(f"replay truncated at {self.p}: need {n}, have {len(self.b) - self.p}")
        s = self.b[self.p : self.p + n]
        self.p += n
        return s

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def names(self) -> list[list[str]]:
        out = []
        for _ in range(2):
            n = self.u32()
            if n > 3:
                raise ValueError(f"bad duelist count {n}")
            out.append([self.take(40).decode("utf-16-le").split("\0", 1)[0] for _ in range(n)])
        return out

    def codes(self) -> list[int]:
        n = self.u32()
        if n > 1000:
            raise ValueError(f"bad card count {n}")
        return list(struct.unpack(f"<{n}I", self.take(4 * n)))

    @property
    def left(self) -> int:
        return len(self.b) - self.p


def _body(data: bytes, want: int) -> tuple[tuple, bytes]:
    if len(data) < _HDR.size:
        raise ValueError("replay shorter than its header")
    h = _HDR.unpack_from(data)
    rtype, _ver, flags, _ts, size, _hash, props = h[:7]
    if rtype != want:
        raise ValueError(f"replay type {rtype:#x}, expected {want:#x}")
    if not flags & F_EXTENDED:
        raise ValueError("old replay header (no 256-bit seed): not supported")
    body = data[_HDR.size :]
    if flags & F_COMPRESSED:
        lc, lp, pb = props[0] % 9, (props[0] // 9) % 5, props[0] // 45
        dict_size = struct.unpack_from("<I", props, 1)[0]
        flt = [{"id": lzma.FILTER_LZMA1, "dict_size": max(dict_size, 4096), "lc": lc, "lp": lp, "pb": pb}]
        body = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=flt).decompress(body, max_length=size)
    if len(body) != size:
        raise ValueError(f"replay body is {len(body)} bytes, header says {size}")
    return h, body


def parse_yrp(data: bytes) -> Yrp:
    h, body = _body(data, YRP1)
    flags = h[2]
    if flags & F_TAG:
        raise ValueError("tag duels are not supported (match record v1 is 1v1)")
    r = _Reader(body)
    names = r.names()
    lp, hand, draw, dflags = r.u32(), r.u32(), r.u32(), r.u64()
    decks = [[(r.codes(), r.codes()) for _ in team] for team in names]
    extra = r.codes()
    responses = []
    while r.left:
        responses.append(r.take(r.u8()))
    return Yrp(tuple(h[8:12]), names, lp, hand, draw, dflags, decks, extra, responses)


def parse_yrpx(data: bytes) -> YrpX:
    h, body = _body(data, YRPX)
    r = _Reader(body)
    out = YrpX(timestamp=h[3], names=r.names(), flags=r.u64())
    while r.left:
        t = r.u8()
        msg = bytes([t]) + r.take(r.u32())
        if t == OLD_REPLAY_FORMAT:
            out.yrp = parse_yrp(msg[1:])
        else:
            out.messages.append(msg)
    if out.yrp is None:
        raise ValueError("no embedded .yrp (message 231): cannot replay")
    return out


def core_messages(x: YrpX) -> list[bytes]:
    """The core's own messages, as our engine's `Duel.trace` records them (server-made messages dropped)."""
    return [m for m in x.messages if m[0] not in SERVER_MSGS]


def to_record(
    x: YrpX,
    engine: Engine,
    *,
    viewer: int | None,
    app: str,
    seats: list[dict] | None = None,
    record_id: str | None = None,
) -> tuple[dict | None, list[str]]:
    """A `full` record (source "yrpx") and the problems found. None + problems when the replay cannot be reproduced
    by our engine either way and order (other scripts than the server's, or an unsupported duel)."""
    y = x.yrp
    assert y is not None
    if any(len(t) != 1 for t in y.decks):
        return None, [f"not a 1v1 duel: {[len(t) for t in y.decks]} duelists"]
    if y.extra_cards:
        return None, [f"extra-rule cards are not supported yet: {y.extra_cards}"]
    # Hint runs follow heap order and differ between processes (NOTEBOOK 2026-10-01): compare canonical traces.
    want = canonical_trace(recorded(core_messages(x)))
    problems: list[str] = []
    for add, order in (("edopro", (0, 1)), ("edopro", (1, 0)), ("ours", (0, 1)), ("ours", (1, 0))):
        rec = {
            "format": MR.FORMAT,
            "id": record_id or MR.uuid7(x.timestamp * 1000),
            "created": _iso(x.timestamp),
            "app": app,
            "engine": MR.engine_info(),
            "cards": {"pack": "server", "sha256": "0" * 64, "server": True},
            "rules": {
                "duel_flags": str(y.flags),
                "start_lp": y.lp,
                "start_hand": y.hand,
                "draw_count": y.draw,
            },
            "kind": "full",
            "engine_seed": [str(w) for w in y.seed],
            "decks": [
                {"main": m, "extra": list(t[0][1]), "hash": MR.deck_hash(m, t[0][1])}
                for t in y.decks
                for m in [list(t[0][0]) if add == "edopro" else edopro_main(t[0][0])]
            ],
            "seats": seats
            or [{"type": "human" if s == viewer else "remote", "client": "unknown"} for s in (0, 1)],
            "mode": "online",
            "responses": [{"seat": -1, "r": r.hex(), "opt": None} for r in y.responses],
            "result": {"winner": None, "turns": 0},
            "source": "yrpx",
        }
        if order != (0, 1):
            rec["add_order"] = list(order)
        if add == "edopro":
            rec["deck_add"] = "edopro"
        try:
            duel, actors = _replay(rec, engine)
        except Exception as e:  # noqa: BLE001 - a failed order is a reason, the other order may still work
            problems.append(f"{add} add, order {order}: replay failed: {type(e).__name__}: {e}")
            continue
        got = canonical_trace(recorded(duel.trace))
        n = min(len(got), len(want))
        if got[:n] != want[:n] or len(got) < len(want) - 1:  # the server may append one MSG_WIN of its own
            k = next((i for i in range(n) if got[i] != want[i]), n)
            problems.append(f"{add} add, order {order}: recorded messages differ at {k} of {len(want)}")
            continue
        for e, s in zip(rec["responses"], actors, strict=True):
            e["seat"] = s
        rec["result"] = _result(duel, x)
        return rec, []
    return None, problems


def _replay(rec: dict, engine: Engine) -> tuple[Duel, list[int]]:
    """Replay `rec` through the core, noting who answered each response (the player of the select message)."""
    duel = Duel(engine, MR.config_of(rec))
    actors = []
    for e in rec["responses"]:
        msg = duel.advance()
        if msg is None:
            raise ValueError(f"duel ended after {len(actors)} of {len(rec['responses'])} responses")
        actors.append(int(msg.player))
        duel.respond(bytes.fromhex(e["r"]))
    duel.advance()
    return duel, actors


def _result(duel: Duel, x: YrpX) -> dict:
    """From the replayed duel, or from the server's own MSG_WIN when the duel ended outside the core (surrender,
    timeout, disconnect: the core never sent MSG_WIN)."""
    winner = duel.winner if duel.winner in (0, 1) else None
    surrendered = False
    if duel.winner is None:
        wins = [m for m in x.messages if m[0] == C.MSG_WIN]
        if wins:
            winner = wins[-1][1] if wins[-1][1] in (0, 1) else None
            surrendered = True
    return {
        "winner": winner,
        "reason": int(duel.win_reason),
        "turns": int(duel.turn),
        "surrendered": surrendered,
        "disconnected": False,
    }


def _iso(ts: int) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- writer (tests, and exporting our games to EDOPro) ---


def _names_bytes(names: list[list[str]]) -> bytes:
    out = b""
    for team in names:
        out += struct.pack("<I", len(team))
        for n in team:
            out += n.encode("utf-16-le")[:40].ljust(40, b"\0")
    return out


def _header(rtype: int, flags: int, ts: int, size: int, props: bytes, seed=(0, 0, 0, 0)) -> bytes:
    return _HDR.pack(rtype, 0, flags, ts, size, 0, props.ljust(8, b"\0"), 1, *seed)


def write_yrpx(y: Yrp, core_msgs: list[bytes], ts: int = 0, compress: bool = True) -> bytes:
    """Serialize like Multirole's Replay::Serialize (MSG_START first, `recorded` core messages, the yrp appended as
    message 231). `y.decks` are in EDOPro's add order (`yrp_decks`)."""
    flags = 0x10 | 0x100 | 0x20 | F_EXTENDED
    yb = _names_bytes(y.names) + struct.pack("<IIIQ", y.lp, y.hand, y.draw, y.flags)
    for team in y.decks:
        for main, extra in team:
            yb += struct.pack(f"<I{len(main)}I", len(main), *main) + struct.pack(
                f"<I{len(extra)}I", len(extra), *extra
            )
    yb += struct.pack(f"<I{len(y.extra_cards)}I", len(y.extra_cards), *y.extra_cards)
    for r in y.responses:
        yb += bytes([len(r)]) + r
    yrp = bytes([OLD_REPLAY_FORMAT]) + _header(YRP1, flags, 0, len(yb), b"", y.seed) + yb
    start = bytes([C.MSG_START, 0]) + struct.pack("<IIHHHH", y.lp, y.lp, 0, 0, 0, 0)
    body = _names_bytes(y.names) + struct.pack("<Q", y.flags)
    for m in [start, *recorded(core_msgs), yrp]:
        body += bytes([m[0]]) + struct.pack("<I", len(m) - 1) + m[1:]
    props = b""
    if compress:
        lc, lp, pb, ds = 3, 0, 2, 1 << 20
        flt = [{"id": lzma.FILTER_LZMA1, "dict_size": ds, "lc": lc, "lp": lp, "pb": pb}]
        comp = lzma.LZMACompressor(lzma.FORMAT_RAW, filters=flt)
        data = comp.compress(body) + comp.flush()
        props = bytes([(pb * 5 + lp) * 9 + lc]) + struct.pack("<I", ds)
        flags |= F_COMPRESSED
    else:
        data = body
    return _header(YRPX, flags, ts, len(body), props) + data


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
