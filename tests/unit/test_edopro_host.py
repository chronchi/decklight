"""EDOPro host tests (each < 5 min with `taskset -c 0-3`).

1. Protocol round trip (structs + split/merged TCP framing).
2. Filter: no bot-hand leak on a full random-vs-random duel + hand-built messages.
3. End to end with an in-process fake EDOPro client over real TCP to duel end.
4. UDP discovery: a HostRequest gets a HostPacket with our port.
"""

import json
import random
import socket
import struct
import threading

import numpy as np
import pytest

from ygo_zero.decks import load_ydk
from ygo_zero.edopro import filter as F
from ygo_zero.edopro import protocol as P
from ygo_zero.edopro.bot import RandomBot
from ygo_zero.edopro.host import DuelHost, check_deck
from ygo_zero.engine import constants as C
from ygo_zero.engine import messages as M
from ygo_zero.engine.actions import make_decision
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.engine.env import PyEnv
from ygo_zero.engine.random_agent import RandomAgent

DECK = "decks/pool/snake-eye.ydk"


# --- 1. protocol round trip ---


def test_protocol_round_trip():
    info = P.HostInfo()
    assert P.HostInfo.unpack(info.pack()) == info
    info.version = P.client_version_bytes(41, 2, 11, 0)  # echo-the-client rule
    assert P.HostInfo.unpack(info.pack()).version == info.version

    v, gid, v2 = P.unpack_join_game(P.pack_join_game(P.PRO_VERSION, 0, info.version))
    assert (v, gid, v2) == (P.PRO_VERSION, 0, info.version)

    mixed = [483, 10000, 20000]
    payload = struct.pack("<II", len(mixed), 0) + struct.pack(f"<{len(mixed)}I", *mixed)
    assert P.parse_update_deck(payload) == (mixed, [])
    with pytest.raises(ValueError):
        P.parse_update_deck(struct.pack("<II", 60, 0) + b"\0" * 8)

    err = P.pack_deck_error(P.DERR_MAINCOUNT, 39, 40, 60, 0)
    assert len(err) == 21 and err[0] == P.ERR_DECKERROR
    assert struct.unpack_from("<BI3II", err) == (P.ERR_DECKERROR, P.DERR_MAINCOUNT, 39, 40, 60, 0)
    assert len(P.build_start(8000, 40, 15, 40, 15, 0)) == 18

    assert P.rps_beats(1, 2) and P.rps_beats(2, 3) and P.rps_beats(3, 1)
    assert not P.rps_beats(2, 1) and not P.rps_beats(1, 1)

    # framing: one frame per byte (split) and three frames at once (merged)
    frames = [(P.CTOS_PLAYER_INFO, P.encode_name("a")), (P.STOC_DUEL_START, b""),
              (P.STOC_GAME_MSG, bytes([C.MSG_WIN, 0, 0]))]
    wire = b"".join(P.pack_frame(p, b) for p, b in frames)
    got = []
    r = P.FrameReader()
    for i in range(len(wire)):
        got += r.feed(wire[i : i + 1])
    assert got == frames
    assert P.FrameReader().feed(wire) == frames
    assert P.decode_name(P.encode_name("ygo-zero")) == "ygo-zero"


# --- 2. filter: no leak of the bot's hidden hand ---


def _play_duel(seed: int):
    engine = Engine()
    deck = load_ydk(DECK)
    duel = engine.new_duel(DuelConfig(deck, deck, seed=seed))
    agents = [RandomAgent(101, engine.cards), RandomAgent(202, engine.cards)]
    while duel.advance() is not None:
        duel.respond(agents[duel.pending.player](duel.pending))
    return duel


def _code_fields_select(raw: bytes) -> list[tuple[int, int]]:
    """(code, controller) of every entry of a SELECT_CARD/TRIBUTE/UNSELECT message."""
    t = raw[0]
    out = []
    if t == C.MSG_SELECT_CARD:
        (count,) = struct.unpack_from("<I", raw, 11)
        pos = 15
        for _ in range(count):
            out.append((struct.unpack_from("<I", raw, pos)[0], raw[pos + 4]))
            pos += 14
    elif t == C.MSG_SELECT_TRIBUTE:
        (count,) = struct.unpack_from("<I", raw, 11)
        pos = 15
        for _ in range(count):
            out.append((struct.unpack_from("<I", raw, pos)[0], raw[pos + 4]))
            pos += 11
    elif t == C.MSG_SELECT_UNSELECT_CARD:
        pos = 12
        for _ in range(2):
            (count,) = struct.unpack_from("<I", raw, pos)
            pos += 4
            for _ in range(count):
                out.append((struct.unpack_from("<I", raw, pos)[0], raw[pos + 4]))
                pos += 14
    return out


def _draw_entries(raw: bytes) -> list[tuple[int, int]]:
    (count,) = struct.unpack_from("<I", raw, 2)
    return [(struct.unpack_from("<I", raw, 6 + 8 * i)[0],
             struct.unpack_from("<I", raw, 10 + 8 * i)[0]) for i in range(count)]


def _shuffle_codes(raw: bytes) -> list[int]:
    (count,) = struct.unpack_from("<I", raw, 2)
    return list(struct.unpack_from(f"<{count}I", raw, 6))


@pytest.mark.parametrize("viewer", [0, 1])
def test_filter_hides_bot_hand(viewer: int):
    bot = 1 - viewer
    duel = _play_duel(7)
    assert duel.ended and duel.winner in (0, 1)
    for raw in duel.trace:
        fm = F.filtered_message(raw, viewer)
        actor = F.waiting_player(raw)
        if actor == bot:
            assert fm is None, f"bot's prompt {raw[0]} leaked to seat {viewer}"
            continue
        t = raw[0]
        if t in (C.MSG_SELECT_CARD, C.MSG_SELECT_TRIBUTE, C.MSG_SELECT_UNSELECT_CARD):
            assert fm is not None  # human's own prompt
            for code, con in _code_fields_select(fm):
                assert con == actor or code == 0, f"uncontrolled code {code} shown"
        elif t in (C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_EXTRA):
            if raw[1] == bot:
                assert fm is not None
                assert _shuffle_codes(fm) == [0] * len(_shuffle_codes(raw))
        elif t == C.MSG_DRAW and raw[1] == bot:
            assert fm is not None
            for (code, _p), (fcode, fpos) in zip(_draw_entries(raw), _draw_entries(fm)):
                assert fcode == (code if fpos & C.POS_FACEUP else 0)
        elif t == C.MSG_MOVE:
            _con, loc, _seq, pos = F._loc(raw, 15)
            hidden = not (loc & (C.LOCATION_GRAVE + C.LOCATION_OVERLAY)) and (
                (loc & (C.LOCATION_DECK + C.LOCATION_HAND)) or (pos & C.POS_FACEDOWN))
            if _con == bot and hidden:
                assert fm is not None and struct.unpack_from("<I", fm, 1)[0] == 0
        elif t == C.MSG_CONFIRM_CARDS:
            (count,) = struct.unpack_from("<I", raw, 2)
            if count and raw[11] in (C.LOCATION_DECK, C.LOCATION_EXTRA):
                assert (fm is None) == (raw[1] == bot)
        elif t in (C.MSG_HINT, C.MSG_CONFIRM_CARDS, C.MSG_WIN) or t not in (
                C.MSG_DRAW, C.MSG_MOVE, C.MSG_SET, C.MSG_SPSUMMONING):
            assert fm is None or fm == raw, f"type {t} must pass through unmodified"

    # refresh encoder: opponent's facedown hand carries no codes; owner's does.
    cards = duel.query_location(bot, C.LOCATION_HAND)
    pub = F.encode_update_data(bot, C.LOCATION_HAND, cards, True)
    (size,) = struct.unpack_from("<I", pub, 3)
    assert size == len(pub) - 7
    n_codes = _count_flag(pub[7:], C.QUERY_CODE)
    n_faceup = sum(1 for c in cards if c is not None and (c.position & C.POS_FACEUP or c.public))
    assert n_codes == n_faceup
    full = F.encode_update_data(bot, C.LOCATION_HAND, cards, False)
    assert _count_flag(full[7:], C.QUERY_CODE) == sum(c is not None for c in cards)


def _count_flag(buf: bytes, flag: int) -> int:
    """Occurrences of a query flag in an encoded record stream."""
    n, pos = 0, 0
    while pos + 2 <= len(buf):
        (size,) = struct.unpack_from("<H", buf, pos)
        if size == 0:
            pos += 2
            continue
        if struct.unpack_from("<I", buf, pos + 2)[0] == flag:
            n += 1
        pos += 2 + size
    return n


def _select_card_raw(player: int, entries: list[tuple[int, int, int, int]]) -> bytes:
    body = struct.pack("<BBIII", player, 0, 1, 2, len(entries))
    for code, con, loc, seq in entries:
        body += struct.pack("<IBBII", code, con, loc, seq, 0)
    return bytes([C.MSG_SELECT_CARD]) + body


def test_filter_hand_built_messages():
    me, op = 0, 1
    raw = _select_card_raw(me, [(1111, me, C.LOCATION_MZONE, 0), (2222, op, C.LOCATION_HAND, 3)])
    fm = F.filtered_message(raw, me)
    assert [c for c, _ in _code_fields_select(fm)] == [1111, 0]
    assert F.filtered_message(raw, op) is None

    draw = bytes([C.MSG_DRAW, op]) + struct.pack("<I", 2)
    draw += struct.pack("<II", 3333, C.POS_FACEDOWN_DEFENSE) + struct.pack("<II", 4444, 0x1)
    fm = F.filtered_message(draw, me)
    assert _draw_entries(fm) == [(0, C.POS_FACEDOWN_DEFENSE), (4444, 0x1)]
    assert F.filtered_message(draw, op) == draw

    move = bytes([C.MSG_MOVE]) + struct.pack("<I", 5555)
    move += struct.pack("<BBII", op, C.LOCATION_HAND, 0, C.POS_FACEDOWN)  # from
    move += struct.pack("<BBII", me, C.LOCATION_GRAVE, 0, C.POS_FACEUP)  # to: public
    assert F.filtered_message(move, me) == move
    move2 = bytes([C.MSG_MOVE]) + struct.pack("<I", 5555)
    move2 += struct.pack("<BBII", me, C.LOCATION_DECK, 0, C.POS_FACEDOWN)
    move2 += struct.pack("<BBII", op, C.LOCATION_HAND, 0, C.POS_FACEDOWN)
    fm = F.filtered_message(move2, me)
    assert struct.unpack_from("<I", fm, 1)[0] == 0 and F.filtered_message(move2, op) == move2

    cc = bytes([C.MSG_CONFIRM_CARDS, op]) + struct.pack("<I", 1) + struct.pack("<IBB", 6666, op,
                                                                               C.LOCATION_DECK)
    assert F.filtered_message(cc, me) is None and F.filtered_message(cc, op) == cc
    cc2 = bytes([C.MSG_CONFIRM_CARDS, op]) + struct.pack("<I", 1) + struct.pack("<IBB", 6666, op,
                                                                                C.LOCATION_GRAVE)
    assert F.filtered_message(cc2, me) == cc2

    assert F.filtered_message(bytes([C.MSG_SET]) + struct.pack("<I", 7777), me)[1:5] == b"\0\0\0\0"
    hint = bytes([C.MSG_HINT, 1, op]) + struct.pack("<Q", 3)
    assert F.filtered_message(hint, me) is None and F.filtered_message(hint, op) == hint
    hint4 = bytes([C.MSG_HINT, 4, me]) + struct.pack("<Q", 3)
    assert F.filtered_message(hint4, op) == hint4 and F.filtered_message(hint4, me) is None

    assert F.waiting_player(bytes([C.MSG_SELECT_IDLECMD, op])) == op
    assert F.waiting_player(bytes([C.MSG_DRAW, op])) is None
    assert F.before_refreshes(bytes([C.MSG_SELECT_IDLECMD, me])) == [
        ("loc", 0, C.LOCATION_MZONE), ("loc", 1, C.LOCATION_MZONE),
        ("loc", 0, C.LOCATION_SZONE), ("loc", 1, C.LOCATION_SZONE),
        ("loc", 0, C.LOCATION_HAND), ("loc", 1, C.LOCATION_HAND)]
    assert F.after_refreshes(bytes([C.MSG_DRAW, op])) == [("loc", op, C.LOCATION_HAND)]


# --- 3. end to end with a fake EDOPro client ---


class FakeClient:
    """Minimal EDOPro client: lobby, RPS/TP, then answers every prompt with random options."""

    def __init__(self, port: int, deck_codes: list[int], pool: set[int], seed: int = 0) -> None:
        self.engine = Engine()
        self.rng = random.Random(seed)
        self.pool = pool
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.reader = P.FrameReader()
        self.deck_codes = deck_codes
        self.start_seen = False
        self.win: tuple[int, int] | None = None
        self.retries = 0
        self.duel_end = False

    def send(self, proto: int, payload: bytes = b"") -> None:
        self.sock.sendall(P.pack_frame(proto, payload))

    def recv(self) -> list[tuple[int, bytes]]:
        data = self.sock.recv(65536)
        if not data:
            raise ConnectionError("server closed")
        return self.reader.feed(data)

    def decide(self, raw: bytes) -> bytes:
        dec = make_decision(M.parse_one(raw), self.engine.cards, lambda: self.pool)
        while True:
            opts = dec.options()
            resp = dec.choose(self.rng.randrange(len(opts)))
            if resp is not None:
                return resp

    def run(self) -> None:
        self.send(P.CTOS_PLAYER_INFO, P.encode_name("fake"))
        self.send(P.CTOS_JOIN_GAME,
                  P.pack_join_game(P.PRO_VERSION, 0, P.client_version_bytes(*P.CLIENT_VERSION)))
        mainc = self.deck_codes
        self.send(P.CTOS_UPDATE_DECK,
                  struct.pack("<II", len(mainc), 0) + struct.pack(f"<{len(mainc)}I", *mainc))
        self.send(P.CTOS_HS_READY)
        while not self.duel_end:
            for proto, payload in self.recv():
                if proto == P.STOC_SELECT_HAND:
                    self.send(P.CTOS_HAND_RESULT, bytes([self.rng.randint(1, 3)]))
                elif proto == P.STOC_SELECT_TP:
                    self.send(P.CTOS_TP_RESULT, bytes([1]))  # human goes first
                elif proto == P.STOC_GAME_MSG:
                    self.on_game(payload)
                elif proto == P.STOC_DUEL_END:
                    self.duel_end = True

    def on_game(self, payload: bytes) -> None:
        t = payload[0]
        if t == C.MSG_START:
            self.start_seen = True
        elif t == C.MSG_WIN:
            self.win = (payload[1], payload[2])
        elif t == C.MSG_RETRY:
            self.retries += 1
        elif t in M.SELECT_TYPES:
            self.send(P.CTOS_RESPONSE, self.decide(payload))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_end_to_end_fake_client(tmp_path):
    try:
        from ygo_zero.engine import inverse  # noqa: F401  (D1; server needs it live)
    except ImportError:
        pytest.skip("D1 engine/inverse.py is missing")

    engine = Engine()
    deck = load_ydk(DECK)
    emb = {int(c) for c in np.load("data/cards/emb/qwen3-0.6b-d512.npz")["codes"]}
    assert check_deck(list(deck.main), list(deck.extra), engine.cards, emb) is None
    mixed = list(deck.main) + list(deck.extra)
    pool = set(mixed)
    host = DuelHost(engine, deck, RandomBot(7), seed=1234, record_root=tmp_path)
    port = _free_port()
    thread = threading.Thread(target=host.serve, args=("127.0.0.1", port, True), daemon=True)
    thread.start()
    client = FakeClient(port, mixed, pool, seed=0)  # seed differs from the bot's: RPS cannot tie forever
    client.run()
    thread.join(timeout=300)

    assert client.start_seen, "client never got MSG_START"
    assert client.win is not None, "duel did not reach MSG_WIN"
    assert client.retries == 0, "server sent MSG_RETRY to a legal client"
    assert client.duel_end, "client never got STOC_DUEL_END"
    assert not thread.is_alive(), "server did not exit after --once duel"

    subs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(subs) == 1 and (subs[0] / "human.ydk").is_file()
    rec = json.loads((subs[0] / "game.jsonl").read_text())
    assert rec["winner"] == client.win[0] and rec["winner"] in (0, 1)
    assert rec["seed"] == 1234 and rec["turns"] > 0 and rec["choices"]
    assert len(rec["actors"]) == len(rec["choices"])

    # the record replays to the same winner
    assert rec["auto_single"] is False
    env = PyEnv.replay(engine, DuelConfig(deck, deck, seed=rec["seed"]), rec["choices"],
                       auto_single=False)
    assert env.done and env.winner == rec["winner"]


# --- 4. UDP discovery ---


def test_udp_discovery():
    engine = Engine()
    host = DuelHost(engine, load_ydk(DECK), RandomBot(0))
    stop = threading.Event()
    port = _free_port()
    thread = threading.Thread(target=host._discovery_loop, args=(stop, port), daemon=True)
    thread.start()
    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", P.DISCOVERY_REPLY_PORT))
        listener.settimeout(5)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(struct.pack("<H", P.NETWORK_CLIENT_ID), ("127.0.0.1", P.DISCOVERY_PORT))
        data, _ = listener.recvfrom(512)
        assert struct.unpack_from("<H", data)[0] == P.NETWORK_SERVER_ID
        assert struct.unpack_from("<H", data, 4)[0] == port
        assert P.decode_name(data[10:50]) == "ygo-zero"
        assert len(data) == 10 + 40 + 63
    finally:
        stop.set()
        thread.join(timeout=5)
