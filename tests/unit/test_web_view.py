"""Unit tests for the web view builders: zone pick labels (item 8) and
vague log lines (item 9). Pure builder tests (no duel needed)."""

import struct
from types import SimpleNamespace

from decklight.engine import constants as C
from decklight.engine.actions import Kind, Option
from decklight.web.view import (
    _HIDDEN_SAY,
    _NEEDS_OBJECT,
    _label,
    _prompt_text,
    build_events,
    zone_label,
)

# Every bit a SelectPlace decision can offer (actions._Place.options ranges).
OFFERABLE = (*range(7), *range(8, 16), *range(16, 23), *range(24, 32))


def _expected(bit: int) -> str:
    """The label for `bit` when the human is asked, derived from the board
    geometry (not from zone_label): own seq counts from the human's left,
    the opponent's board is mirrored, EMZ/pendulum halves are left/right."""
    rel, z = bit >> 4, bit & 15
    mine = rel == 0
    who = "Your" if mine else "Opponent's"
    if z < 8:  # monster zone (5 main + 2 EMZ)
        if z >= 5:
            side = "left" if (z == 5) == mine else "right"
            return f"Extra Monster Zone ({side})"
        col = z + 1 if mine else 5 - z
        return f"{who} Monster Zone {col}"
    seq = z - 8  # spell/trap zone (5 backrow + field + 2 pendulum)
    if seq <= 4:
        col = seq + 1 if mine else 5 - seq
        return f"{who} Spell & Trap Zone {col}"
    if seq == 5:
        return f"{who} Field Zone"
    side = "left" if (seq == 6) == mine else "right"
    return f"{who} Pendulum Zone ({side})"


def test_zone_label_every_offerable_bit_both_seats():
    for human in (0, 1):
        for bit in OFFERABLE:
            got = zone_label(bit, human, human)
            assert got == _expected(bit), (human, bit, got)
            assert "Place in zone" not in got and str(bit) not in got.split(), (human, bit, got)


def test_zone_label_spot_words():
    assert zone_label(2, 0, 0) == "Your Monster Zone 3"
    assert zone_label(5, 0, 0) == "Extra Monster Zone (left)"
    assert zone_label(6, 0, 0) == "Extra Monster Zone (right)"
    assert zone_label(22, 0, 0) == "Extra Monster Zone (left)"  # opp seq 6, shared left cell
    assert zone_label(16, 0, 0) == "Opponent's Monster Zone 5"  # opp seq 0 mirrors to the right
    assert zone_label(20, 0, 0) == "Opponent's Monster Zone 1"
    assert zone_label(9, 0, 0) == "Your Spell & Trap Zone 2"
    assert zone_label(13, 1, 1) == "Your Field Zone"
    assert zone_label(29, 0, 0) == "Opponent's Field Zone"
    assert zone_label(14, 0, 0) == "Your Pendulum Zone (left)"
    assert zone_label(15, 1, 1) == "Your Pendulum Zone (right)"


def test_place_option_label_flows_through_label_fn():
    for human in (0, 1):
        for bit in OFFERABLE:
            opt = Option(Kind.PLACE, arg=bit)
            assert _label(opt, None, None, None, human, human) == _expected(bit)
    # Without seats (old callers): the legacy fallback stays.
    assert _label(Option(Kind.PLACE, arg=8), None, None, None) == "Place in zone 8"


def _trace_hint(seat: int, code: int) -> bytes:
    return bytes([C.MSG_HINT, C.HINT_SELECTMSG, seat]) + struct.pack("<Q", code)


def test_place_prompt_text_names_the_card():
    from decklight.web.view import DB

    code = next(iter(DB.names()))
    name = DB.names()[code]
    env = SimpleNamespace(duel=SimpleNamespace(trace=[_trace_hint(0, code)]))
    assert _prompt_text(env, 0, "place", {}, None) == f"Choose a zone for {name}"
    env = SimpleNamespace(duel=SimpleNamespace(trace=[]))
    assert _prompt_text(env, 0, "place", {}, None) == "Choose a zone"


# --- item 9: an empty card name renders the generic hidden wording ---


def _hist(kind: Kind, code: int, actor: int = 0, public: bool = True):
    return SimpleNamespace(row=[int(kind), -1, code], actor=actor, public=public)


def test_empty_name_uses_hidden_wording():
    env = SimpleNamespace(history=[_hist(k, 0) for k in _NEEDS_OBJECT])
    for e in build_events(env, 0):
        kind = Kind[e["kind"].upper()]
        want = _HIDDEN_SAY.get(kind, "makes a hidden move")
        assert e["text"] == want[:1].upper() + want[1:], (e["kind"], e["text"])


def test_empty_name_keeps_complete_sentences():
    env = SimpleNamespace(
        history=[
            _hist(Kind.PASS, 0),
            _hist(Kind.TO_BATTLE, 0),
            _hist(Kind.TO_END, 0),
            _hist(Kind.NUMBER, 0),  # "declares the number" needs no card
        ]
    )
    texts = [e["text"] for e in build_events(env, 0)]
    assert texts == ["Passes", "Enters the Battle Phase", "Ends the turn", "Declares the number"]


def test_named_events_unchanged():
    from decklight.web.view import DB

    code = next(iter(DB.names()))
    name = DB.names()[code]
    env = SimpleNamespace(history=[_hist(Kind.SELECT, code), _hist(Kind.ATTACK, code)])
    texts = [e["text"] for e in build_events(env, 0)]
    assert texts == [f"Selects {name}", f"Attacks with {name}"]
