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


@dataclass(frozen=True)
class ProtocolConfig:
    """Wire versions for one end of the EDOPro LAN protocol (plan §3.3).

    `pro_version` is the u16 CTOS_JOIN_GAME.version slot; `client_version` is the
    4-byte ClientVersion (EDOPro major, EDOPro minor, core major, core minor).
    Defaults are today's values, so host and client agree unless overridden
    (--pro-version / --client-version, or WindBot's Version=).
    """

    pro_version: int = PRO_VERSION
    client_version: tuple[int, int, int, int] = CLIENT_VERSION

    @staticmethod
    def parse_version(s: str) -> tuple[int, int, int, int]:
        """'41.0.11.0' -> (41, 0, 11, 0). Raises ValueError on a bad string."""
        parts = s.split(".")
        if len(parts) != 4:
            raise ValueError(f"client version needs 4 dotted parts, got {s!r}")
        try:
            v = tuple(int(p) for p in parts)
        except ValueError:
            raise ValueError(f"client version needs integers, got {s!r}") from None
        if any(not 0 <= x <= 255 for x in v):
            raise ValueError(f"client version parts must be 0-255, got {s!r}")
        return v  # type: ignore[return-value]

    @classmethod
    def from_strings(cls, pro: str, client: str) -> "ProtocolConfig":
        """'0x1354' (or '4948') + '41.0.11.0' -> ProtocolConfig."""
        return cls(pro_version=int(pro, 0), client_version=cls.parse_version(client))

    def version_int(self) -> int:
        """WindBot-style packed int: major | minor<<8 | core_major<<16 | core_minor<<24."""
        a, b, c, d = self.client_version
        return a | (b << 8) | (c << 16) | (d << 24)

    @classmethod
    def from_windbot_version(cls, v: int) -> "ProtocolConfig":
        """WindBot's `Version=` int -> ProtocolConfig reproducing WindBot's wire bytes.

        WindBot packs the EDOPro+core version into one int (default
        40|1<<8|10<<16 = EDOPro 40.1, core 10.0) and sends its low 16 bits in
        the u16 version slot and the full 4 bytes LE as version2. EDOPro proper
        sends PRO_VERSION (0x1354) in the u16 slot instead; servers
        (EDOPro generic_duel, Multirole) only check version2, so both connect.
        """
        if not 0 <= v <= 0xFFFFFFFF:
            raise ValueError(f"WindBot Version out of u32 range: {v!r}")
        b = v.to_bytes(4, "little")
        return cls(pro_version=v & 0xFFFF, client_version=(b[0], b[1], b[2], b[3]))

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
STOC_CHANGE_SIDE = 0x7
STOC_WAITING_SIDE = 0x8
STOC_CREATE_GAME = 0x11
STOC_JOIN_GAME = 0x12
STOC_TYPE_CHANGE = 0x13
STOC_LEAVE_GAME = 0x14
STOC_DUEL_START = 0x15
STOC_DUEL_END = 0x16
STOC_REPLAY = 0x17
STOC_TIME_LIMIT = 0x18
STOC_CHAT = 0x19
STOC_HS_PLAYER_ENTER = 0x20
STOC_HS_PLAYER_CHANGE = 0x21
STOC_HS_WATCH_CHANGE = 0x22

MODE_SINGLE = 0x0

PLAYERCHANGE_READY = 0x9
PLAYERCHANGE_NOTREADY = 0xA

ERR_JOINERROR = 0x1
ERR_DECKERROR = 0x2
ERR_SIDEERROR = 0x3
ERR_VERERROR = 0x4
ERR_VERERROR2 = 0x5

# JoinError::JERR_TYPE (network.h)
JERR_UNABLE = 0
JERR_PASSWORD = 1
JERR_REFUSED = 2
_JERR_NAMES = ("unable to join", "wrong password", "join refused")

# DeckError::DERR_TYPE (network.h)
DERR_NONE = 0
DERR_LFLIST = 1
DERR_OCGONLY = 2
DERR_TCGONLY = 3
DERR_UNKNOWNCARD = 4
DERR_CARDCOUNT = 5
DERR_MAINCOUNT = 6
DERR_EXTRACOUNT = 7
DERR_SIDECOUNT = 8
DERR_FORBTYPE = 9
DERR_UNOFFICIALCARD = 10
DERR_INVALIDSIZE = 11
DERR_TOOMANYLEGENDS = 12
DERR_TOOMANYSKILLS = 13
_DERR_NAMES = (
    "no error", "forbidden-list", "ocg-only", "tcg-only", "unknown card", "card count",
    "main count", "extra count", "side count", "forbidden type", "unofficial card",
    "invalid size", "too many legends", "too many skills",
)

# STOC_HS_PLAYER_CHANGE states (WindBot PlayerChange; host.py sends READY/NOTREADY)
HS_OBSERVE = 0x8
HS_READY = 0x9
HS_NOTREADY = 0xA
HS_LEAVE = 0xB
_HS_NAMES = {HS_OBSERVE: "observe", HS_READY: "ready", HS_NOTREADY: "not ready", HS_LEAVE: "leave"}


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


def pack_join_game(version: int, gameid: int, version2: bytes, password: str = "") -> bytes:
    """CTOS_JOIN_GAME: u16 version, u32 gameid, u16 pass[20], ClientVersion (52 bytes).

    `password` is the room password as UTF-16 (WindBot `HostInfo=`); empty by
    default, which keeps the old 3-arg call working (pass = 20 NUL units).
    """
    assert len(version2) == 4
    return struct.pack("<HI", version, gameid) + encode_name(password, 20) + version2


def unpack_join_game(payload: bytes) -> tuple[int, int, bytes]:
    """CTOS_JOIN_GAME payload -> (version, gameid, version2)."""
    version, gameid = struct.unpack_from("<HI", payload)
    version2 = bytes(payload[46:50])
    return version, gameid, version2


def pack_player_info(name: str) -> bytes:
    """CTOS_PLAYER_INFO payload: u16 name[20]."""
    return encode_name(name, 20)


def pack_update_deck(main: list[int], extra: list[int], side: list[int]) -> bytes:
    """CTOS_UPDATE_DECK payload: u32 mainc (= main + extra), u32 sidec, then codes.

    Round-trips through `parse_update_deck` (-> (main + extra, side)).
    """
    codes = list(main) + list(extra) + list(side)
    out = struct.pack("<II", len(main) + len(extra), len(side))
    if codes:
        out += struct.pack(f"<{len(codes)}I", *codes)
    return out


def pack_hand_choice(res: int) -> bytes:
    """CTOS_HAND_RESULT payload: the RPS throw (1 rock / 2 scissors / 3 paper)."""
    if res not in (1, 2, 3):
        raise ValueError(f"RPS throw must be 1/2/3, got {res!r}")
    return bytes([res])


def pack_tp_choice(first: bool) -> bytes:
    """CTOS_TP_RESULT payload: 1 = go first, 0 = go second."""
    return bytes([1 if first else 0])


def pack_response(data: bytes) -> bytes:
    """CTOS_RESPONSE payload: the raw core response (already encoded)."""
    return bytes(data)


def pack_time_confirm() -> bytes:
    """CTOS_TIME_CONFIRM payload: empty."""
    return b""


def parse_hand_result(payload: bytes) -> tuple[int, int]:
    """STOC_HAND_RESULT payload -> (ours, theirs); the server orders per recipient."""
    if len(payload) < 2:
        raise ValueError(f"hand result too short: {len(payload)}")
    return payload[0], payload[1]


def parse_tp_result(payload: bytes) -> bool:
    """STOC_TP_RESULT payload -> True when we go first."""
    if len(payload) < 1:
        raise ValueError("tp result is empty")
    return payload[0] == 1


def pack_hand_result(res1: int, res2: int) -> bytes:
    return bytes([res1, res2])


def parse_join_game(payload: bytes) -> HostInfo:
    """STOC_JOIN_GAME payload -> HostInfo (raises ValueError when short)."""
    if len(payload) < _HOSTINFO_FMT.size:
        raise ValueError(f"join-game packet too short: {len(payload)}")
    return HostInfo.unpack(payload)


def parse_type_change(payload: bytes) -> tuple[int, bool]:
    """STOC_TYPE_CHANGE payload -> (seat, is_host)."""
    if len(payload) < 1:
        raise ValueError("type-change packet is empty")
    t = payload[0]
    return t & 0xF, bool((t >> 4) & 0xF)


def parse_hs_player_enter(payload: bytes) -> tuple[str, int]:
    """STOC_HS_PLAYER_ENTER payload -> (name, seat)."""
    if len(payload) < 41:
        raise ValueError(f"player-enter packet too short: {len(payload)}")
    return decode_name(payload[:40]), payload[40]


def parse_hs_player_change(payload: bytes) -> tuple[int, str]:
    """STOC_HS_PLAYER_CHANGE payload -> (seat, state name)."""
    if len(payload) < 1:
        raise ValueError("player-change packet is empty")
    status = payload[0]
    return (status >> 4) & 0xF, _HS_NAMES.get(status & 0xF, f"state {status & 0xF}")


def parse_time_limit(payload: bytes) -> tuple[int, int]:
    """STOC_TIME_LIMIT payload -> (player, seconds left)."""
    if len(payload) < 3:
        raise ValueError(f"time-limit packet too short: {len(payload)}")
    return payload[0], struct.unpack_from("<H", payload, 1)[0]


def parse_error_msg(payload: bytes) -> str:
    """STOC_ERROR_MSG payload -> readable text (join/deck/side/version errors)."""
    if not payload:
        return "empty error message"
    etype = payload[0]
    if etype == ERR_JOINERROR:
        (code,) = struct.unpack_from("<I", payload, 1) if len(payload) >= 5 else (0,)
        name = _JERR_NAMES[code] if code < len(_JERR_NAMES) else f"code {code}"
        return f"join error: {name}"
    if etype == ERR_DECKERROR:
        if len(payload) < 21:
            return "deck error: (truncated)"
        (_t, current, minimum, maximum, code) = struct.unpack_from("<5I", payload, 1)
        name = _DERR_NAMES[_t] if _t < len(_DERR_NAMES) else f"type {_t}"
        msg = f"deck error: {name} (current {current}, allowed {minimum}-{maximum})"
        if _t == DERR_UNKNOWNCARD:
            msg += f" card {code}"
        return msg
    if etype == ERR_SIDEERROR:
        (code,) = struct.unpack_from("<I", payload, 1) if len(payload) >= 5 else (0,)
        return f"side-deck error: code {code}"
    if etype in (ERR_VERERROR, ERR_VERERROR2):
        ver = bytes(payload[4:8]) if len(payload) >= 8 else b""
        want = ".".join(str(b) for b in ver) if ver else "?"
        return f"version error: server wants client version {want}"
    return f"unknown error type {etype}"


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
