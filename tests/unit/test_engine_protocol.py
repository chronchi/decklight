import struct

from ygo_zero.decks import parse_ydk
from ygo_zero.engine import _gen_constants
from ygo_zero.engine import constants as C
from ygo_zero.engine import messages as M
from ygo_zero.engine import responses as R
from ygo_zero.engine.announce import is_declarable
from ygo_zero.engine.duel import seed_words
from ygo_zero.engine.random_agent import RandomAgent, sum_is_valid
from ygo_zero.engine.resources import CardData


def test_constants_match_pinned_header():
    expected = _gen_constants.render(_gen_constants.parse(_gen_constants.HEADER.read_text()))
    assert _gen_constants.OUT.read_text() == expected, "run: uv run python -m ygo_zero.engine._gen_constants"


def test_octal_link_markers():
    assert C.LINK_MARKER_LEFT == 0x8 and C.LINK_MARKER_TOP_RIGHT == 0x100


def test_parse_ydk():
    deck = parse_ydk("#created by x\n#main\n1\n1\n2\n#extra\n3\n!side\n4\n\n")
    assert (deck.main, deck.extra, deck.side) == ((1, 1, 2), (3,), (4,))


def test_split_buffer():
    a, b = bytes([C.MSG_NEW_TURN, 1]), bytes([C.MSG_NEW_PHASE, 4, 0])
    buf = struct.pack("<I", 2) + a + struct.pack("<I", 0) + struct.pack("<I", 3) + b
    assert M.split_buffer(buf) == [a, b]
    assert M.parse_one(b).phase == 4


def _card_short(code, con, loc, seq):
    return struct.pack("<IBBB", code, con, loc, seq)


def test_parse_select_counter():
    raw = struct.pack("<BBHHI", C.MSG_SELECT_COUNTER, 1, 0x1009, 3, 2)
    raw += _card_short(10, 1, C.LOCATION_MZONE, 2) + struct.pack("<H", 2)
    raw += _card_short(11, 0, C.LOCATION_SZONE, 0) + struct.pack("<H", 5)
    msg = M.parse_one(raw)
    assert isinstance(msg, M.SelectCounter)
    assert (msg.player, msg.counter_type, msg.count) == (1, 0x1009, 3)
    assert msg.cards == [
        (M.CardRef(10, 1, C.LOCATION_MZONE, 2), 2),
        (M.CardRef(11, 0, C.LOCATION_SZONE, 0), 5),
    ]
    counts = struct.unpack("<2h", RandomAgent(0).respond(msg))
    assert sum(counts) == 3 and counts[0] <= 2


def test_parse_announces_and_rps():
    race = M.parse_one(
        struct.pack("<BBBQ", C.MSG_ANNOUNCE_RACE, 0, 2, C.RACE_DRAGON | C.RACE_ZOMBIE | C.RACE_FIEND)
    )
    resp = struct.unpack("<Q", RandomAgent(1).respond(race))[0]
    assert resp.bit_count() == 2 and resp & ~race.available == 0
    attr = M.parse_one(
        struct.pack("<BBBI", C.MSG_ANNOUNCE_ATTRIB, 1, 1, C.ATTRIBUTE_DARK | C.ATTRIBUTE_LIGHT)
    )
    assert struct.unpack("<I", RandomAgent(2).respond(attr))[0] in (C.ATTRIBUTE_DARK, C.ATTRIBUTE_LIGHT)
    rps = M.parse_one(bytes([C.MSG_ROCK_PAPER_SCISSORS, 1]))
    assert rps.player == 1 and 1 <= struct.unpack("<i", RandomAgent(3).respond(rps))[0] <= 3


def test_select_sum_exact_and_at_least():
    ref = M.CardRef(1, 0, C.LOCATION_HAND, 0)
    # exact: levels 4, 4, 3 (3 can also count as 4); target 8 with 2 cards
    msg = M.SelectSum(C.MSG_SELECT_SUM, b"", 0, False, 8, 1, 2, [], [(ref, 4), (ref, 4), (ref, 3 | 4 << 16)])
    assert sum_is_valid(msg, [0, 1]) and sum_is_valid(msg, [0, 2]) and not sum_is_valid(msg, [0])
    picked = struct.unpack_from("<iI", RandomAgent(0).respond(msg))
    assert picked == (2, 2)
    # at least 5 with no redundant card: {4, 4} ok, {4, 4, 3} is redundant
    msg = M.SelectSum(C.MSG_SELECT_SUM, b"", 0, True, 5, 0, 0, [], [(ref, 4), (ref, 4), (ref, 3)])
    assert sum_is_valid(msg, [0, 1]) and not sum_is_valid(msg, [0, 1, 2]) and not sum_is_valid(msg, [0])


def test_responses():
    assert R.command(R.IDLE_ACTIVATE, 3) == struct.pack("<i", 3 << 16 | 5)
    assert R.cards([0, 2]) == struct.pack("<iI", 2, 2) + b"\x00\x02"
    assert R.place([(1, C.LOCATION_MZONE, 2)]) == bytes([1, 4, 2])


def test_announce_card_filter():
    cd = CardData(46986414, 0, (0x10A2,), C.TYPE_MONSTER | C.TYPE_NORMAL, 7, C.ATTRIBUTE_DARK,
                  C.RACE_SPELLCASTER, 2500, 2100, 0, 0, 0)  # fmt: skip
    assert is_declarable(cd, [C.TYPE_MONSTER, C.OPCODE_ISTYPE])
    assert not is_declarable(cd, [C.TYPE_SPELL, C.OPCODE_ISTYPE])
    assert is_declarable(cd, [0xA2, C.OPCODE_ISSETCARD, C.ATTRIBUTE_DARK, C.OPCODE_ISATTRIBUTE, C.OPCODE_AND])
    assert not is_declarable(cd, [0x10A2 + 1, C.OPCODE_ISSETCARD])


def test_seed_words_nonzero_and_distinct():
    assert all(seed_words(0)) and seed_words(1) != seed_words(2)
