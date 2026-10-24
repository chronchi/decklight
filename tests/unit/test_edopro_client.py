"""EDOPro network client tests (each < 5 min with `taskset -c 0-3`).

1. Protocol: client packers round-trip against the host-side parsers; version
   parsing; error-message decoding.
2. WindBot launch contract: Key=Value parsing, serve-URL parsing, catalog.
3. Loopback: ClientSession (RandomAgent + DecisionOnlyView) joins our DuelHost
   and plays full duels: same winner on both sides, 0 retries, DUEL_END seen.
4. Timeout: a server that accepts and never answers -> exit code 4.
"""

import json
import socket
import struct
import threading

import pytest

from decklight.decks import load_ydk
from decklight.edopro import bots as B
from decklight.edopro import protocol as P
from decklight.edopro.bot import RandomBot
from decklight.edopro.client import OK, TIMEOUT, ClientSession, DecisionOnlyView
from decklight.edopro.host import DuelHost
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv
from decklight.play.agents import RandomAgent

DECK = "decks/pool/snake-eye.ydk"


# --- 1. protocol ---


def test_client_packers_round_trip():
    # join with password: version2 stays at [46:50], password is UTF-16 at [6:46]
    payload = P.pack_join_game(P.PRO_VERSION, 7, P.client_version_bytes(*P.CLIENT_VERSION), "secret")
    assert len(payload) == 50
    assert P.unpack_join_game(payload) == (P.PRO_VERSION, 7, P.client_version_bytes(*P.CLIENT_VERSION))
    assert P.decode_name(payload[6:46]) == "secret"
    # old 3-arg call still works (empty password)
    assert P.unpack_join_game(P.pack_join_game(P.PRO_VERSION, 0, bytes([41, 0, 11, 0]))) == (
        P.PRO_VERSION, 0, bytes([41, 0, 11, 0]))

    # update deck round-trips through the host-side parser
    assert P.parse_update_deck(P.pack_update_deck([1, 2], [3], [4])) == ([1, 2, 3], [4])
    assert P.parse_update_deck(P.pack_update_deck([], [], [])) == ([], [])

    assert P.pack_player_info("me") == P.encode_name("me", 20)
    assert P.pack_hand_choice(2) == bytes([2])
    with pytest.raises(ValueError):
        P.pack_hand_choice(4)
    assert P.pack_tp_choice(True) == bytes([1]) and P.pack_tp_choice(False) == bytes([0])
    assert P.pack_response(b"\x01\x02") == b"\x01\x02"
    assert P.pack_time_confirm() == b""

    # STOC parsers
    assert P.parse_join_game(P.HostInfo().pack()) == P.HostInfo()
    assert P.parse_type_change(bytes([0x10])) == (0, True)
    assert P.parse_hs_player_enter(P.encode_name("ab") + bytes([2])) == ("ab", 2)
    assert P.parse_hs_player_change(bytes([(1 << 4) | P.PLAYERCHANGE_READY])) == (1, "ready")
    assert P.parse_time_limit(bytes([1, 30, 0])) == (1, 30)
    assert P.parse_hand_result(bytes([3, 1])) == (3, 1)
    assert P.parse_tp_result(bytes([1])) is True


def test_version_parsing():
    assert P.ProtocolConfig.parse_version("41.0.11.0") == (41, 0, 11, 0)
    with pytest.raises(ValueError):
        P.ProtocolConfig.parse_version("41.0.11")
    with pytest.raises(ValueError):
        P.ProtocolConfig.parse_version("41.0.11.999")
    assert P.ProtocolConfig.from_strings("0x1354", "41.0.11.0") == P.ProtocolConfig()
    assert P.ProtocolConfig.from_strings("4948", "41.0.11.0") == P.ProtocolConfig()
    # WindBot default: EDOPro 40.1, core 10.0; low16 goes in the u16 slot
    w = P.ProtocolConfig.from_windbot_version(40 | 1 << 8 | 10 << 16)
    assert (w.pro_version, w.client_version) == (296, (40, 1, 10, 0))
    assert P.ProtocolConfig().version_int() == 41 | 11 << 16


def test_error_message_decoding():
    assert "password" in P.parse_error_msg(P.pack_join_error(P.JERR_PASSWORD))
    assert "refused" in P.parse_error_msg(P.pack_join_error(P.JERR_REFUSED))
    text = P.parse_error_msg(P.pack_deck_error(P.DERR_MAINCOUNT, 39, 40, 60, 0))
    assert "main count" in text and "39" in text and "40-60" in text
    text = P.parse_error_msg(P.pack_deck_error(P.DERR_UNKNOWNCARD, 0, 0, 0, 1234))
    assert "unknown card" in text and "1234" in text
    text = P.parse_error_msg(bytes([P.ERR_VERERROR2, 0, 0, 0, 41, 0, 11, 0]))
    assert "41.0.11.0" in text
    assert P.parse_error_msg(b"") != ""


# --- 2. WindBot launch contract ---


def test_kv_parsing():
    kv, rest = B.parse_kv_args(["Name=test", "Deck=random-snake-eye", "Host=127.0.0.1",
                                "Port=7922", "--debug"])
    assert kv == {"name": "test", "deck": "random-snake-eye", "host": "127.0.0.1", "port": 7922}
    assert rest == ["--debug"]
    kv, _ = B.parse_kv_args(["HostInfo=secret", "Version=655656", "Hand=2",
                             "Debug=true", "Chat=no"])
    assert kv == {"password": "secret", "version": 655656, "hand": 2,
                  "debug": True, "chat": False}
    with pytest.raises(ValueError):
        B.parse_kv_args(["Bogus=1"])
    with pytest.raises(ValueError):
        B.parse_kv_args(["Port=abc"])


def test_serve_query_parsing():
    q = B.parse_serve_query("name=t&deck=random-snake-eye&host=127.0.0.1&port=7924"
                            "&version=720937&password=x&hand=1&chat=true")
    assert q == {"name": "t", "deck": "random-snake-eye", "host": "127.0.0.1",
                 "port": 7924, "password": "x", "version": 720937, "hand": 1, "chat": True}
    assert B.parse_serve_query("name=t&host=h&port=1") == {"name": "t", "host": "h", "port": 1}


def test_catalog_loading_and_resolution(tmp_path):
    catalog = B.load_catalog()
    assert "random-snake-eye" in catalog and "raw-blueeyes" in catalog
    entry = catalog["random-snake-eye"]
    assert entry.agent == "random" and entry.deck.endswith("snake-eye.ydk")
    assert catalog["raw-blueeyes"].agent == "raw"
    text = B.list_bots_text(catalog)
    assert "random-snake-eye" in text and "raw-blueeyes" in text

    path, spec = B.resolve_deck("random-snake-eye", catalog)
    assert spec is not None and path is not None and path.is_file()
    assert spec.agent == "random"
    path, spec = B.resolve_deck("decks/pool/snake-eye.ydk", catalog)
    assert path is not None and path.is_file()  # bare .ydk -> on-the-fly default bundle
    assert spec is not None and spec.agent == "random" and spec.deck_path() == path
    path, spec = B.resolve_deck("decks/pool/snake-eye", catalog)  # suffix optional
    assert path is not None and path.is_file() and spec is not None
    assert B.resolve_deck("no-such-bot", catalog) == (None, None)

    (tmp_path / "mine.toml").write_text(
        'deck = "decks/pool/blue-eyes.ydk"\nagent = "random"\n')
    mine = B.load_catalog(tmp_path)
    assert mine["mine"].agent == "random"  # name falls back to the stem


# --- 3 + 4. loopback and timeout ---


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run_loopback(deal_seed: int, client_seed: int, tmp_path) -> tuple[ClientSession, dict]:
    engine = Engine()
    deck = load_ydk(DECK)
    host = DuelHost(engine, deck, RandomBot(deal_seed), seed=deal_seed, record_root=tmp_path)
    port = _free_port()
    thread = threading.Thread(target=host.serve, args=("127.0.0.1", port, True), daemon=True)
    thread.start()
    session = ClientSession(
        "127.0.0.1", port, "loopback", deck, RandomAgent(client_seed),
        view_factory=lambda seat, codes: DecisionOnlyView(seat, codes, cards=engine.cards),
        timeout_s=120.0,
    )
    code = session.run()
    thread.join(timeout=300)
    assert not thread.is_alive(), "server did not exit after the --once duel"
    subs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(subs) == 1
    rec = json.loads((subs[0] / "game.jsonl").read_text())
    return session, {"code": code, "record": rec}


def _run_stream_loopback(deal_seed: int, client_seed: int, tmp_path) -> tuple[ClientSession, dict]:
    """Same as `_run_loopback` but the client plays through `StreamView` (the real client path)."""
    from decklight.play.stream import StreamView

    engine = Engine()
    deck = load_ydk(DECK)
    host = DuelHost(engine, deck, RandomBot(deal_seed), seed=deal_seed, record_root=tmp_path)
    port = _free_port()
    thread = threading.Thread(target=host.serve, args=("127.0.0.1", port, True), daemon=True)
    thread.start()
    main, extra = list(deck.main), list(deck.extra)
    session = ClientSession(
        "127.0.0.1", port, "loopback", deck, RandomAgent(client_seed),
        view_factory=lambda seat, codes: StreamView({"main": main, "extra": extra},
                                                    engine.cards, seat=seat),
        timeout_s=120.0,
    )
    code = session.run()
    thread.join(timeout=300)
    assert not thread.is_alive(), "server did not exit after the --once duel"
    subs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(subs) == 1
    rec = json.loads((subs[0] / "game.jsonl").read_text())
    return session, {"code": code, "record": rec}


@pytest.mark.slow  # ~35 s per full random-vs-random duel over the real client view
def test_loopback_full_duel_stream_view(tmp_path):
    try:
        from decklight.engine import inverse  # noqa: F401 (D1; server needs it live)
    except ImportError:
        pytest.skip("D1 engine/inverse.py is missing")
    session, out = _run_stream_loopback(14, 4, tmp_path)
    assert out["code"] == OK, f"client exited with {out['code']}"
    assert session.record.duel_end, "client never saw STOC_DUEL_END"
    assert session.record.winner is not None, "client never saw MSG_WIN"
    assert session.record.retries == 0, "server sent MSG_RETRY to a legal client"
    assert session.record.seat in (0, 1)
    rec = out["record"]
    assert rec["winner"] == session.record.winner and rec["winner"] in (0, 1)


@pytest.mark.slow  # ~35 s per full random-vs-random duel; 3 seeds ≈ 2 min
@pytest.mark.parametrize("deal_seed,client_seed", [(11, 1), (12, 2), (13, 3)])
def test_loopback_full_duel(deal_seed, client_seed, tmp_path):
    try:
        from decklight.engine import inverse  # noqa: F401 (D1; server needs it live)
    except ImportError:
        pytest.skip("D1 engine/inverse.py is missing")
    session, out = _run_loopback(deal_seed, client_seed, tmp_path)
    assert out["code"] == OK, f"client exited with {out['code']}"
    assert session.record.duel_end, "client never saw STOC_DUEL_END"
    assert session.record.winner is not None, "client never saw MSG_WIN"
    assert session.record.retries == 0, "server sent MSG_RETRY to a legal client"
    assert session.record.seat in (0, 1)
    rec = out["record"]
    assert rec["winner"] == session.record.winner[0] and rec["winner"] in (0, 1)
    assert rec["seed"] == deal_seed and rec["turns"] > 0 and rec["choices"]
    # the host record replays to the same winner
    engine = Engine()
    deck = load_ydk(DECK)
    env = PyEnv.replay(engine, DuelConfig(deck, deck, seed=rec["seed"]), rec["choices"],
                       auto_single=False)
    assert env.done and env.winner == rec["winner"]


def test_rps_tie_guard_trips():
    session = ClientSession(
        "127.0.0.1", 1, "tie", ([89631139] * 40, [], []), RandomAgent(0),
        view_factory=lambda seat, codes: DecisionOnlyView(seat, codes, cards={}),
        timeout_s=1.0,
    )
    for _ in range(49):
        assert not session._note_hand_result(bytes([2, 2]))
    assert session._note_hand_result(bytes([2, 2]))  # 50th tie trips
    session._note_hand_result(bytes([1, 2]))  # a decided round resets
    assert not session._note_hand_result(bytes([2, 2]))


def test_timeout_when_server_stays_silent():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def accept_only():
        conn, _ = srv.accept()
        try:
            conn.recv(1024)  # read the join, then never answer
            threading.Event().wait(30)
        finally:
            conn.close()

    thread = threading.Thread(target=accept_only, daemon=True)
    thread.start()
    try:
        from decklight.engine.duel import Engine as _E
        cards = _E().cards
    except (ImportError, OSError):
        cards = {}
    session = ClientSession(
        "127.0.0.1", port, "silent", ([89631139] * 40, [], []), RandomAgent(0),
        view_factory=lambda seat, codes: DecisionOnlyView(seat, codes, cards=cards),
        timeout_s=2.0,
    )
    assert session.run() == TIMEOUT
    srv.close()


def test_decision_only_view_singletons():
    from decklight.engine import constants as C

    view = DecisionOnlyView(0, [1, 2, 3], cards={}, pool={1, 2, 3})
    assert not view.needs_answer and view.opts == [] and view.take_auto() is None
    with pytest.raises(NotImplementedError):
        view.obs()
    # non-select traffic is ignored; the winner is remembered
    view.feed(bytes([C.MSG_DRAW, 0]) + struct.pack("<I", 0))
    assert not view.needs_answer
    view.feed(bytes([C.MSG_WIN, 1, 0]))
    assert view.winner == (1, 0)
