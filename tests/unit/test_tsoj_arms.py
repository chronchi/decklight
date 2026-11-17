"""tsoj-arms: forced chain-0 arm, and the client/protocol fixes tsoj's windbot needs."""

import struct
import types

import numpy as np

from decklight.edopro import arms as A
from decklight.edopro import filter as F
from decklight.edopro import protocol as P
from decklight.edopro.client import ClientSession
from decklight.engine import constants as C
from decklight.engine import inverse
from decklight.engine import responses as R
from decklight.engine.actions import Kind, Option, make_decision
from decklight.engine.messages import CardRef, SortCards
from decklight.play.agents import RandomAgent


def _obs(turn_player_is_me=0, chain=0, mask=(True, True, True)):
    g = np.zeros(24, np.int32)
    g[2] = 4  # turn
    g[5] = turn_player_is_me
    g[20] = chain
    return {"global": g, "action_mask": np.array(mask, bool)}


PASS = Option(Kind.PASS)
ACT_A = Option(Kind.ACTIVATE, None, 111)
ACT_B = Option(Kind.ACTIVATE, None, 222)


def test_parse_arm():
    assert A.parse_arm("p") == A.ArmSpec("p", 0)
    assert A.parse_arm("top@3") == A.ArmSpec("top", 3)
    assert A.parse_arm("rand@1") == A.ArmSpec("rand", 1)
    for bad in ("top", "top@0", "rand@x", "nope@1"):
        try:
            A.parse_arm(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should be rejected")


def test_window_filter():
    opts = [PASS, ACT_A, ACT_B]
    ok, acts = A.is_window(_obs(), opts, 0)
    assert ok and acts == [1, 2]
    assert not A.is_window(_obs(turn_player_is_me=1), opts, 0)[0]  # our turn
    assert not A.is_window(_obs(chain=1), opts, 0)[0]  # chain not empty
    assert not A.is_window(_obs(mask=(True, False, False)), opts, 0)[0]  # no activation
    assert not A.is_window(_obs(), opts, 1)[0]  # P itself activates


class _Policy:
    def __init__(self, scores):
        self.scores = np.asarray(scores, float)

    def logits(self, batch):
        return self.scores[None, :]


class _Inner:
    def __init__(self, scores):
        self.policy = _Policy(scores)

    def rps(self):
        return 1

    def go_first(self):
        return True

    def act(self, view):
        return int(np.argmax(self.policy.scores))


class _View:
    def __init__(self, obs, opts, seat=0):
        self._obs, self.opts, self.seat = obs, opts, seat

    def obs(self):
        return self._obs


def test_top_arm_forces_exactly_once(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "_name", lambda code: f"c{code}")
    inner = _Inner([5.0, 1.0, 0.0])  # P would pass; top activation is index 1
    agent = A.ArmedAgent(inner, A.ArmSpec("top", 2), seed=0, game_index=0,
                         log_path=tmp_path / "arms.jsonl")
    view = _View(_obs(), [PASS, ACT_A, ACT_B])
    assert agent.act(view) == 0  # first window: P
    assert agent.act(view) == 1  # second window: forced top
    assert agent.act(view) == 0  # after the force P plays on
    assert agent.windows == 3 and agent.forced
    rec = types.SimpleNamespace(n_responses=3, winner=(0, 2), turns=7)
    agent.finalize(rec, deck="skydeck")
    row = __import__("json").loads((tmp_path / "arms.jsonl").read_text())
    assert row["arm"] == "top" and row["k"] == 2 and row["forced"] is True
    assert row["card"] == "c111" and row["p_choice"] == "PASS"
    assert row["n_activations"] == 2 and row["turn"] == 4


def test_arm_logs_forced_false_when_no_window(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "_name", lambda code: str(code))
    agent = A.ArmedAgent(_Inner([5.0, 1.0, 0.0]), A.ArmSpec("rand", 1), log_path=tmp_path / "a.jsonl")
    agent.act(_View(_obs(turn_player_is_me=1), [PASS, ACT_A, ACT_B]))
    agent.finalize(types.SimpleNamespace(n_responses=1, winner=None, turns=2))
    row = __import__("json").loads((tmp_path / "a.jsonl").read_text())
    assert row["forced"] is False and row["decision"] == -1


def test_parse_join_game_accepts_windbots_short_hostinfo():
    short = P.HostInfo().pack()[:56]
    info = P.parse_join_game(short)
    assert info.start_lp == 8000 and info.main_max == 0  # missing tail fields read 0
    try:
        P.parse_join_game(b"\0" * 20)
    except ValueError:
        pass
    else:
        raise AssertionError("a 20-byte HostInfo must be rejected")


def test_filter_confirm_own_extra_and_hand_to_deck_cost():
    me, op = 0, 1
    # our own Extra Deck card revealed to us (this was hidden before)
    own = bytes([C.MSG_CONFIRM_CARDS, me]) + struct.pack("<I", 1) + struct.pack(
        "<IBB", 7777, me, C.LOCATION_EXTRA) + struct.pack("<I", 0)
    assert F.filtered_message(own, me) == own
    assert F.filtered_message(own, op) is None
    # a hand card returned to the Deck as a cost is shown to the opponent too
    move = bytes([C.MSG_MOVE]) + struct.pack("<I", 5555)
    move += struct.pack("<BBII", op, C.LOCATION_HAND, 0, C.POS_FACEDOWN)
    move += struct.pack("<BBII", op, C.LOCATION_DECK, 0, C.POS_FACEDOWN)
    move += struct.pack("<I", C.REASON_COST)
    assert F.filtered_message(move, me) == move


def test_inverse_single_card_sort():
    msg = SortCards(C.MSG_SORT_CARD, b"", 0, [CardRef(111, 0, C.LOCATION_DECK, 0)])
    dec = make_decision(msg)
    assert isinstance(dec, __import__("decklight.engine.actions", fromlist=["_Sort"])._Sort)
    resp = R.sort([0])
    assert inverse._sort_path(dec, False, resp) == [0]


def test_lobby_accepts_msg_start_without_duel_start(tmp_path):
    deck = types.SimpleNamespace(main=[], extra=[], side=[])
    sess = ClientSession("127.0.0.1", 1, "P", deck, RandomAgent(0),
                         view_factory=lambda seat, codes: None, record_dir=tmp_path,
                         lobby_timeout_s=None)
    frames = [
        (P.STOC_JOIN_GAME, P.HostInfo().pack()),
        (P.STOC_TYPE_CHANGE, bytes([0])),
        (P.STOC_GAME_MSG, bytes([C.MSG_START, 0])),
    ]
    sess._wait = lambda: frames
    sent: list[int] = []
    sess.send = lambda proto, payload=b"": sent.append(proto)
    code, pending = sess._lobby()
    assert code is None
    assert pending == frames[2:]
    assert P.CTOS_UPDATE_DECK in sent and P.CTOS_HS_READY in sent
