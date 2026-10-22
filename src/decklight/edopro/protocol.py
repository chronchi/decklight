"""EDOPro LAN protocol (gframe/network.h, netserver.cpp): packet ids, structs, framing.

Framing (netserver.cpp ServerEchoRead): u16 LE length (proto byte + payload),
then u8 proto, then payload. One TCP read can hold a split or several merged frames.
"""

import struct
from dataclasses import dataclass

NETWORK_SERVER_ID = 0x7428
NETWORK_CLIENT_ID = 0xDEF6
SERVER_HANDSHAKE = 4043399681
PRO_VERSION = 0x1354  # edopro gframe/game.cpp
CLIENT_VERSION = (41, 0, 11, 0)  # EDOPro 41.0, core 11.0 (gframe/config.h)

DISCOVERY_PORT = 7920
DISCOVERY_REPLY_PORT = 7921

CTOS_RESPONSE = 0x1
CTOS_UPDATE_DECK = 0x2
CTOS_HAND_RESULT = 0x3
CTOS_TP_RESULT = 0x4
CTOS_PLAYER_INFO = 0x10
CTOS_CREATE_GAME = 0x11
CTOS_JOIN_GAME = 0x12
CTOS_LEAVE_GAME = 0x13
CTOS_SURRENDER = 0x14
CTOS_TIME_CONFIRM = 0x15
CTOS_CHAT = 0x16
CTOS_HS_TODUELIST = 0x20
CTOS_HS_TOOBSERVER = 0x21
CTOS_HS_READY = 0x22
CTOS_HS_NOTREADY = 0x23
CTOS_HS_KICK = 0x24
CTOS_HS_START = 0x25

STOC_GAME_MSG = 0x1
STOC_ERROR_MSG = 0x2
STOC_SELECT_HAND = 0x3
STOC_SELECT_TP = 0x4
STOC_HAND_RESULT = 0x5
STOC_TP_RESULT = 0x6
STOC_CREATE_GAME = 0x11
STOC_JOIN_GAME = 0x12
STOC_TYPE_CHANGE = 0x13
STOC_DUEL_START = 0x15
STOC_DUEL_END = 0x16
STOC_REPLAY = 0x17
STOC_HS_PLAYER_ENTER = 0x20
STOC_HS_PLAYER_CHANGE = 0x21
STOC_HS_WATCH_CHANGE = 0x22

MODE_SINGLE = 0x0

PLAYERCHANGE_READY = 0x9
PLAYERCHANGE_NOTREADY = 0xA

ERR_JOINERROR = 0x1
ERR_DECKERROR = 0x2

# DeckError::DERR_TYPE (network.h)
DERR_NONE = 0
DERR_UNKNOWNCARD = 4
DERR_MAINCOUNT = 6
DERR_EXTRACOUNT = 7
DERR_INVALIDSIZE = 11


def encode_name(s: str, units: int = 20) -> bytes:
    """UTF-16LE name field, NUL-padded (BufferIO::EncodeUTF16 semantics)."""
    b = s[:units].encode("utf-16-le")
    return b + b"\0" * (units * 2 - len(b))


def decode_name(b: bytes) -> str:
    return b.decode("utf-16-le").split("\0")[0]


def client_version_bytes(major: int, minor: int, core_major: int, core_minor: int) -> bytes:
    return bytes([major, minor, core_major, core_minor])


_HOSTINFO_FMT = struct.Struct("<IBBBBBIBBHII4siiiIIH6H")
assert _HOSTINFO_FMT.size == 63


@dataclass
class HostInfo:
    lflist: int = 0
    rule: int = 0
    mode: int = MODE_SINGLE
    duel_rule: int = 5
    no_check_deck_content: int = 0
    no_shuffle_deck: int = 0
    start_lp: int = 8000
    start_hand: int = 5
    draw_count: int = 1
    time_limit: int = 0
    duel_flag_high: int = 0
    handshake: int = SERVER_HANDSHAKE
    version: bytes = client_version_bytes(*CLIENT_VERSION)
    team1: int = 1
    team2: int = 1
    best_of: int = 1
    duel_flag_low: int = 0
    forbiddentypes: int = 0
    extra_rules: int = 0
    main_min: int = 40
    main_max: int = 60
    extra_min: int = 0
    extra_max: int = 15
    side_min: int = 0
    side_max: int = 15

    def pack(self) -> bytes:
        return _HOSTINFO_FMT.pack(
            self.lflist, self.rule, self.mode, self.duel_rule, self.no_check_deck_content,
            self.no_shuffle_deck, self.start_lp, self.start_hand, self.draw_count, self.time_limit,
            self.duel_flag_high, self.handshake, self.version, self.team1, self.team2,
            self.best_of, self.duel_flag_low, self.forbiddentypes, self.extra_rules,
            self.main_min, self.main_max, self.extra_min, self.extra_max, self.side_min, self.side_max,
        )

    @classmethod
    def unpack(cls, b: bytes) -> "HostInfo":
        f = _HOSTINFO_FMT.unpack(b[: _HOSTINFO_FMT.size])
        return cls(
            lflist=f[0], rule=f[1], mode=f[2], duel_rule=f[3], no_check_deck_content=f[4],
            no_shuffle_deck=f[5], start_lp=f[6], start_hand=f[7], draw_count=f[8],
            time_limit=f[9], duel_flag_high=f[10], handshake=f[11], version=f[12],
            team1=f[13], team2=f[14], best_of=f[15], duel_flag_low=f[16],
            forbiddentypes=f[17], extra_rules=f[18], main_min=f[19], main_max=f[20],
            extra_min=f[21], extra_max=f[22], side_min=f[23], side_max=f[24],
        )


def pack_join_game(version: int, gameid: int, version2: bytes) -> bytes:
    """CTOS_JOIN_GAME: u16 version, u32 gameid, u16 pass[20], ClientVersion (52 bytes)."""
    assert len(version2) == 4
    return struct.pack("<HI", version, gameid) + encode_name("", 20) + version2


def unpack_join_game(payload: bytes) -> tuple[int, int, bytes]:
    """CTOS_JOIN_GAME payload -> (version, gameid, version2)."""
    version, gameid = struct.unpack_from("<HI", payload)
    version2 = bytes(payload[46:50])
    return version, gameid, version2


def pack_hand_result(res1: int, res2: int) -> bytes:
    return bytes([res1, res2])


def pack_deck_error(derr_type: int, current: int = 0, minimum: int = 0,
                    maximum: int = 0, code: int = 0) -> bytes:
    """STOC_ERROR_MSG payload for a DECKERROR (network.h DeckError layout)."""
    return struct.pack("<BI3II", ERR_DECKERROR, derr_type, current, minimum, maximum, code)


def pack_join_error(code: int) -> bytes:
    return struct.pack("<BI", ERR_JOINERROR, code)


def parse_update_deck(payload: bytes) -> tuple[list[int], list[int]]:
    """CTOS_UPDATE_DECK payload -> (main_plus_extra codes, side codes).

    Layout (menu_handler.cpp UpdateDeck): u32 mainc (= main + extra), u32 sidec,
    then main, extra, side codes. Raises ValueError on a short buffer.
    """
    if len(payload) < 8:
        raise ValueError(f"deck packet too short: {len(payload)}")
    mainc, sidec = struct.unpack_from("<II", payload)
    if 8 + 4 * (mainc + sidec) > len(payload):
        raise ValueError(f"deck packet claims {mainc}+{sidec} cards but holds {len(payload)} bytes")
    codes = list(struct.unpack_from(f"<{mainc + sidec}I", payload, 8)) if mainc + sidec else []
    return codes[:mainc], codes[mainc:]


def build_start(lp: int, deck0: int, extra0: int, deck1: int, extra1: int, seat: int) -> bytes:
    """MSG_START for one recipient (generic_duel.cpp TPResult): seat = recipient's engine seat."""
    return struct.pack("<BBIIHHHH", 4, seat, lp, lp, deck0, extra0, deck1, extra1)


def pack_frame(proto: int, payload: bytes = b"") -> bytes:
    return struct.pack("<HB", len(payload) + 1, proto) + payload


class FrameReader:
    """Accumulates TCP reads into (proto, payload) frames; handles split and merged reads."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self.buf += data
        out = []
        while len(self.buf) >= 2:
            (n,) = struct.unpack_from("<H", self.buf)
            if len(self.buf) < 2 + n:
                break
            frame = bytes(self.buf[2 : 2 + n])
            del self.buf[: 2 + n]
            if frame:
                out.append((frame[0], frame[1:]))
        return out


def rps_beats(a: int, b: int) -> bool:
    """1 rock beats 2 scissors beats 3 paper beats 1 (generic_duel.cpp HandResult)."""
    return (a, b) in ((1, 2), (2, 3), (3, 1))
