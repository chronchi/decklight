import struct

from ygo_zero.engine import constants as C
from ygo_zero.engine import messages as M
from ygo_zero.engine import responses as R
from ygo_zero.engine.actions import Kind, make_decision
from ygo_zero.engine.messages import CardRef
from ygo_zero.engine.random_agent import sum_is_valid


def _ref(i, loc=C.LOCATION_MZONE, con=0):
    return CardRef(1000 + i, con, loc, i)


def _kinds(dec):
    return [o.kind for o in dec.options()]


def test_select_card_min_max():
    msg = M.SelectCard(C.MSG_SELECT_CARD, b"", 0, False, 1, 2, [_ref(i) for i in range(3)])
    dec = make_decision(msg)
    assert _kinds(dec) == [Kind.SELECT] * 3  # no FINISH before min, not cancelable
    assert dec.choose(2) is None
    assert _kinds(dec) == [Kind.SELECT, Kind.SELECT, Kind.FINISH]
    assert dec.selected == [_ref(2)]
    assert dec.choose(0) == R.cards([0, 2])  # auto-sent at max, ascending


def test_select_card_cancel_and_finish():
    msg = M.SelectCard(C.MSG_SELECT_CARD, b"", 0, True, 0, 2, [_ref(i) for i in range(2)])
    dec = make_decision(msg)
    assert _kinds(dec) == [Kind.SELECT, Kind.SELECT, Kind.FINISH, Kind.CANCEL]
    assert dec.choose(3) == R.CANCEL
    assert make_decision(msg).choose(2) == R.cards([])


def test_tribute_offers_only_completable_cards():
    # need 2 release points, max 2 cards; params: 1, 1, 2
    cards = [(_ref(0), 1), (_ref(1), 1), (_ref(2), 2)]
    msg = M.SelectTribute(C.MSG_SELECT_TRIBUTE, b"", 0, False, 2, 2, cards)
    dec = make_decision(msg)
    assert _kinds(dec) == [Kind.SELECT] * 3
    assert dec.choose(2) is None  # the 2-pointer alone reaches min
    assert _kinds(dec) == [Kind.SELECT, Kind.SELECT, Kind.FINISH]
    assert dec.choose(2) == R.cards([2])
    # max 1 card: only the 2-pointer can complete
    msg1 = M.SelectTribute(C.MSG_SELECT_TRIBUTE, b"", 0, False, 2, 1, cards)
    assert [o.card for o in make_decision(msg1).options()] == [_ref(2)]


def test_select_sum_exact_offers_valid_extensions():
    cards = [(_ref(i), lv) for i, lv in enumerate([4, 4, 3, 8])]
    msg = M.SelectSum(C.MSG_SELECT_SUM, b"", 0, False, 8, 1, 2, [], cards)
    dec = make_decision(msg)
    assert [o.card.sequence for o in dec.options()] == [0, 1, 3]  # the level 3 fits no subset
    assert dec.choose(0) is None
    assert [o.kind for o in dec.options()] == [Kind.SELECT]  # only the other 4
    assert dec.choose(0) is None
    assert _kinds(dec) == [Kind.FINISH]
    resp = dec.choose(0)
    assert resp == R.cards([0, 1]) and sum_is_valid(msg, [0, 1])


def test_select_sum_must_cards_count_toward_acc_not_max():
    must = [(_ref(9), 4)]
    cards = [(_ref(i), lv) for i, lv in enumerate([4, 2])]
    msg = M.SelectSum(C.MSG_SELECT_SUM, b"", 0, False, 8, 1, 1, must, cards)
    dec = make_decision(msg)
    assert [o.card.sequence for o in dec.options()] == [0]
    assert dec.selected == [_ref(9)]


def test_place_multi_step():
    flag = ~((1 << 2) | (1 << 3) | (1 << 17)) & 0xFFFFFFFF  # my MZ 2,3 and opp MZ 1 free
    msg = M.SelectPlace(C.MSG_SELECT_DISFIELD, b"", 1, 2, flag)
    dec = make_decision(msg)
    assert [o.arg for o in dec.options()] == [2, 3, 17]
    assert dec.choose(2) is None
    assert [o.arg for o in dec.options()] == [2, 3]
    assert dec.choose(0) == R.place([(0, C.LOCATION_MZONE, 1), (1, C.LOCATION_MZONE, 2)])


def test_sort_gives_ranks():
    msg = M.SortCards(C.MSG_SORT_CARD, b"", 0, [_ref(i) for i in range(3)])
    dec = make_decision(msg)
    assert dec.choose(2) is None  # card 2 first
    assert dec.choose(1) == R.sort([2, 1, 0])  # then card 1 (index 1 of [0, 1]); card 0 last → ranks


def test_counter_and_bits():
    msg = M.SelectCounter(C.MSG_SELECT_COUNTER, b"", 0, 0x1009, 2, [(_ref(0), 1), (_ref(1), 3)])
    dec = make_decision(msg)
    assert dec.choose(0) is None
    assert [o.card for o in dec.options()] == [_ref(1)]
    assert dec.choose(0) == struct.pack("<2h", 1, 1)
    race = M.AnnounceRace(C.MSG_ANNOUNCE_RACE, b"", 0, 2, C.RACE_WARRIOR | C.RACE_DRAGON | C.RACE_YOKAI)
    dec = make_decision(race)
    assert [o.arg for o in dec.options()] == [0, 13, 62]
    dec.choose(2)
    assert dec.choose(0) == R.announce_race(C.RACE_WARRIOR | C.RACE_YOKAI)


def test_chain_pass_and_unselect_finish():
    ch = M.ChainOption(_ref(0), 5 << 20 | 1, 0)
    msg = M.SelectChain(C.MSG_SELECT_CHAIN, b"", 0, 0, False, 0, 0, [ch])
    dec = make_decision(msg)
    assert _kinds(dec) == [Kind.ACTIVATE, Kind.PASS]
    assert dec.options()[0].desc == 5 << 20 | 1
    msg = M.SelectUnselectCard(C.MSG_SELECT_UNSELECT_CARD, b"", 0, True, True, 1, 2, [_ref(0)], [_ref(1)])
    dec = make_decision(msg)
    assert _kinds(dec) == [Kind.SELECT, Kind.UNSELECT, Kind.FINISH]
    assert dec.choose(1) == R.unselect_card(1)
