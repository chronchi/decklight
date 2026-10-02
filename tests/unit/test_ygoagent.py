"""Encoder unit tests for the ygo-agent 0546 gauntlet opponent (hand-built obs rows)."""

import numpy as np
import pytest

from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.eval import ygoagent as Y


def _first_codes(n=4):
    try:
        with open("data/models/code_list.txt") as f:
            return [int(line.split()[0]) for line in f][:n]
    except FileNotFoundError:
        return [101101001, 101101002, 101101003, 101101004]  # encoder tests use codes() anyway


CODES = _first_codes()
KNOWN, KNOWN2 = CODES[0], CODES[1]
UNKNOWN = 123_456_789  # not a real card code


def card_row(
    code=KNOWN,
    loc=2,
    seq=1,
    ctrl=0,
    pos=0,
    public=1,
    type_=0x21,
    attr=0x1,
    race=0x2000,
    level=4,
    atk=1800,
    dfn=1200,
    status=0,
):
    r = np.zeros(21, np.int32)
    r[:] = [
        code,
        loc,
        seq,
        ctrl,
        0,
        pos,
        public,
        type_,
        attr,
        race,
        level,
        0,
        0,
        0,
        atk,
        dfn,
        atk,
        dfn,
        0,
        status,
        0,
    ]
    return r


def codes() -> dict[int, int]:
    return {0: 0, KNOWN: 1, KNOWN2: 2}


def test_load_code_ids_two_column_and_blanks(tmp_path):
    p = tmp_path / "codes.txt"
    p.write_text(f"\n{KNOWN} 1\n\n{KNOWN2} 0\n")
    ids = Y.load_code_ids(p)
    assert ids == {0: 0, KNOWN: 1, KNOWN2: 2}


def test_vintage_code_ids():
    ids = Y.vintage_code_ids()
    assert ids[0] == 0 and len(ids) == 865
    assert sorted(ids.values()) == list(range(865))
    assert ids[Y._VINTAGE_CODES[0]] == 1 and ids[Y._VINTAGE_CODES[-1]] == 864
    assert len(set(Y._VINTAGE_CODES)) == 864


def test_vintage_ids_are_in_vocab_range():
    ids = Y.vintage_code_ids()
    assert max(ids.values()) == 864  # and 864 < 1000: interpreter gathers OOB at >= 1000


@pytest.mark.skipif(not Y.CODE_LIST_PATH.exists(), reason="staged code list absent")
def test_vintage_codes_subset_of_staged_list():
    staged = set(Y.load_code_ids(Y.CODE_LIST_PATH)) - {0}
    assert set(Y._VINTAGE_CODES) <= staged


def test_vintage_validation_rejects_wrong_list(tmp_path):
    p = tmp_path / "codes.txt"
    p.write_text("12345\n")
    with pytest.raises(ValueError):
        Y.vintage_code_ids(p)


def test_encode_cards_nonvintage_code_is_unknown():
    vintage = set(Y._VINTAGE_CODES)
    other = next(c for c in CODES if c not in vintage) if any(c not in vintage for c in CODES) else UNKNOWN
    cards = np.zeros((160, 21), np.int32)
    cards[0] = card_row(code=other)
    enc, _ = Y.encode_cards(cards, Y.vintage_code_ids())
    assert (enc[0][0], enc[0][1]) == (0, 0)


def test_scorable_options_filters_cancel_and_unselect():
    kinds = np.array([Kind.SELECT, Kind.CANCEL], np.int32)
    assert Y.scorable_options(C.MSG_SELECT_UNSELECT_CARD, kinds, 2).tolist() == [True, False]
    assert Y.scorable_options(C.MSG_SELECT_CARD, kinds, 2).tolist() == [True, False]
    kinds = np.array([Kind.SELECT, Kind.UNSELECT, Kind.FINISH], np.int32)
    assert Y.scorable_options(C.MSG_SELECT_UNSELECT_CARD, kinds, 3).tolist() == [True, False, True]
    kinds = np.array([Kind.ACTIVATE, Kind.PASS], np.int32)
    assert Y.scorable_options(C.MSG_SELECT_CHAIN, kinds, 2).tolist() == [True, True]
    # Degenerate (nothing scorable): everything scores rather than argmax over -inf.
    kinds = np.array([Kind.CANCEL], np.int32)
    assert Y.scorable_options(C.MSG_SELECT_CARD, kinds, 1).tolist() == [True]


def test_transforms():
    assert Y.int_transform(1) == (0, 1)
    assert Y.int_transform(256 + 7) == (1, 7)
    assert Y.float_transform(8000) == (31, 64)
    assert Y.float_transform(-2) == (255, 254)  # same modulo semantics as features.py


def test_effect_to_id():
    assert Y.effect_to_id(Kind.ACTIVATE, 0, 0) == 1  # desc 0 on an effectful kind
    assert Y.effect_to_id(Kind.SUMMON, 0, 0) == 0  # structural kinds carry no effect
    assert Y.effect_to_id(Kind.ACTIVATE, 0, 1050) == Y.SYSTEM_STRING_TO_ID[1050]
    assert Y.effect_to_id(Kind.ACTIVATE, KNOWN, 3) == 5  # card string idx -> idx + 2
    assert Y.effect_to_id(Kind.YES, KNOWN, 0) == 2
    assert Y.effect_to_id(Kind.ACTIVATE, 0, 999_999) == 0  # unknown system string: N/A, no crash


def test_encode_cards_monster():
    cards = np.zeros((160, 21), np.int32)
    cards[5] = card_row()
    enc, index = Y.encode_cards(cards, codes())
    assert enc.shape == (160, 41) and enc.dtype == np.uint8
    assert list(index[:3]) == [1, 2, 3]
    row = enc[5]
    assert (row[0], row[1]) == (0, 1)  # code id 1 split to 2 bytes
    assert row[2] == 2 and row[3] == 0 and row[4] == 0  # hand: loc 2, no seq, me
    assert row[5] == 0 and row[6] == 0  # no position / not overlay
    assert row[7] == 1 and row[8] == 14  # earth, dragon
    assert row[9] == 4 and row[10] == 0 and row[11] == 0  # level, no counter, not negated
    assert (row[12], row[13]) == (7, 8) and (row[14], row[15]) == (4, 176)  # 1800 / 1200
    assert row[16] == 1 and row[20] == 1 and row[16:41].sum() == 2  # monster + effect
    assert enc[6].sum() == 0  # padding row stays zero


def test_encode_cards_mzone_overlay_hidden_unknown():
    cards = np.zeros((160, 21), np.int32)
    cards[0] = card_row(loc=3, seq=2, pos=1)  # my mzone, face-up attack
    cards[1] = card_row(loc=8, seq=3)  # overlay unit on my zone 2
    cards[2] = card_row(
        code=0, loc=2, seq=1, ctrl=1, public=0, type_=0, attr=0, race=0, level=0, atk=0, dfn=0
    )  # hidden opp hand card
    cards[3] = card_row(code=UNKNOWN)  # not in the code list
    cards[4] = card_row(level=0)  # xyz-style: rank folds into level
    cards[4][11] = 4
    cards[5] = card_row(status=C.STATUS_DISABLED)  # negated
    enc, _ = Y.encode_cards(cards, codes())
    assert enc[0][2] == 3 and enc[0][3] == 2 and enc[0][5] == 1
    assert enc[1][2] == 3 and enc[1][5] == 5 and enc[1][6] == 1  # overlay -> mzone/faceup/flag
    assert (enc[2][0], enc[2][1]) == (0, 0) and enc[2][2] == 2 and enc[2][4] == 1
    assert enc[2][7] == 0 and enc[2][8] == 0  # hidden stats stay zero
    assert (enc[3][0], enc[3][1]) == (0, 0)  # unknown code -> id 0
    assert enc[4][9] == 4
    assert enc[5][11] == 1


def test_encode_global():
    g = np.array(
        [
            8000,
            100,
            99,
            C.PHASE_MAIN1,
            1,
            1,
            38,
            5,
            0,
            0,
            0,
            0,
            14,
            41,
            5,
            0,
            0,
            0,
            0,
            15,
            0,
            C.MSG_SELECT_IDLECMD,
            0,
            1,
        ],
        np.int32,
    )
    x = Y.encode_global(g)
    assert x.shape == (23,) and x.dtype == np.uint8
    assert (x[0], x[1]) == (31, 64) and (x[2], x[3]) == (0, 100)
    assert x[4] == 16 and x[5] == 2 and x[6] == 1 and x[7] == 1  # turn clamped, main1, first, my turn
    assert x[8] == 38 and x[14] == 14 and x[15] == 41 and x[21] == 15
    assert x[22] == 0


def opt(kind, row=0, code=KNOWN, ec=0, ei=0, arg=0, msg=C.MSG_SELECT_IDLECMD, loc=2, ctrl=0, step=0):
    return [int(kind), row, code, ec, ei, arg, msg, loc, ctrl, step]


def test_encode_actions_kinds():
    acts = np.zeros((64, 10), np.int32)
    acts[0] = opt(Kind.SUMMON, row=3)
    acts[1] = opt(Kind.ATTACK, arg=1, msg=C.MSG_SELECT_BATTLECMD)
    acts[2] = opt(Kind.ATTACK, arg=0, msg=C.MSG_SELECT_BATTLECMD)
    acts[3] = opt(Kind.TO_BATTLE)
    acts[4] = opt(Kind.FINISH, row=-1, code=0, msg=C.MSG_SELECT_CARD, loc=0, ctrl=-1)
    acts[5] = opt(Kind.PASS, row=-1, code=0, msg=C.MSG_SELECT_CHAIN, loc=0, ctrl=-1)
    acts[6] = opt(Kind.ACTIVATE, ec=KNOWN, ei=3, msg=C.MSG_SELECT_CHAIN)
    index = np.arange(1, 161)
    x = Y.encode_actions(acts, 7, index, codes())
    assert x.shape == (24, 12)
    assert x[0][0] == 4 and x[0][3] == 1 and x[0][4] == 4  # spec 1-based, idle, summon
    assert x[1][4] == 7 and x[2][4] == 6  # direct vs normal attack
    assert x[3][4] == 0 and x[3][7] == 1  # to battle: no act, battle phase
    assert x[4][0] == 0 and x[4][5] == 1  # finish flag, no card
    assert x[5][3] == 2 and x[5][4] == 9  # chain pass -> cancel
    assert x[6][6] == 5  # card string idx 3 -> 5


def test_encode_actions_place_position_number_attribute():
    acts = np.zeros((64, 10), np.int32)
    acts[0] = opt(Kind.PLACE, row=-1, code=0, arg=0, msg=C.MSG_SELECT_PLACE, loc=0, ctrl=-1)
    acts[1] = opt(Kind.PLACE, row=-1, code=0, arg=8, msg=C.MSG_SELECT_PLACE, loc=0, ctrl=-1)
    acts[2] = opt(Kind.PLACE, row=-1, code=0, arg=16, msg=C.MSG_SELECT_PLACE, loc=0, ctrl=-1)
    acts[3] = opt(Kind.PLACE, row=-1, code=0, arg=24, msg=C.MSG_SELECT_PLACE, loc=0, ctrl=-1)
    acts[4] = opt(
        Kind.POSITION, row=-1, code=0, arg=C.POS_FACEUP_DEFENSE, msg=C.MSG_SELECT_POSITION, loc=0, ctrl=-1
    )
    acts[5] = opt(Kind.NUMBER, row=-1, code=0, arg=99, msg=C.MSG_ANNOUNCE_NUMBER, loc=0, ctrl=-1)
    acts[6] = opt(
        Kind.ATTRIBUTE, row=-1, code=0, arg=C.ATTRIBUTE_LIGHT, msg=C.MSG_ANNOUNCE_ATTRIB, loc=0, ctrl=-1
    )
    x = Y.encode_actions(acts, 7, np.arange(1, 161), codes())
    assert x[0][10] == 1 and x[1][10] == 8 and x[2][10] == 16 and x[3][10] == 23
    assert x[4][8] == 4 and x[5][9] == 12 and x[6][11] == 5


def test_encode_actions_truncates_at_24():
    acts = np.zeros((64, 10), np.int32)
    for i in range(30):
        acts[i] = opt(Kind.SUMMON, row=i)
    x = Y.encode_actions(acts, 30, np.arange(1, 161), codes())
    assert x.shape == (24, 12) and x[23][0] == 24  # slot 24 = option 24; option 25+ never scored


def test_encode_history_order_and_turn():
    hist = np.zeros((32, 13), np.int32)
    mask = np.zeros(32, bool)
    # most recent first: [0] 1 turn ago, [1] 5 turns ago.
    hist[0] = opt(Kind.SUMMON, ec=0, ei=0)[:10] + [0, 1, C.PHASE_MAIN1]
    hist[1] = opt(Kind.ACTIVATE, ec=KNOWN, ei=3, msg=C.MSG_SELECT_CHAIN)[:10] + [1, 5, C.PHASE_END]
    mask[:2] = True
    x = Y.encode_history(hist, mask, codes())
    assert x.shape == (32, 14)
    assert x[0][4] == 8 and x[0][12] == 5 and x[0][13] == 9  # oldest first: activate, end
    assert x[1][4] == 4 and x[1][12] == 1 and x[1][13] == 2
    assert x[0][0] == 0 and (x[0][1], x[0][2]) == (0, 1)  # no spec pointer; card id kept
    assert x[2].sum() == 0


def test_encode_history_unmapped_msg_zeroes_turn():
    hist = np.zeros((32, 13), np.int32)
    mask = np.zeros(32, bool)
    hist[0] = opt(Kind.RPS, row=-1, code=0, arg=1, msg=C.MSG_ROCK_PAPER_SCISSORS, loc=0, ctrl=-1)[:10] + [
        0,
        0,
        C.PHASE_DRAW,
    ]
    mask[0] = True
    x = Y.encode_history(hist, mask, codes())
    assert x[0][3] == 0 and x[0][12] == 0  # mirrors their padding rule for msg == 0


MODEL = Y.MODEL_PATH


@pytest.mark.skipif(not MODEL.exists(), reason="model file absent (data/ is gitignored)")
def test_policy_forward_on_synthetic_row():
    ai_edge = pytest.importorskip("ai_edge_litert.interpreter")
    assert ai_edge is not None
    pol = Y.YgoAgentPolicy(str(MODEL), "data/models/code_list.txt", seed=0)
    assert pol.name == "ygoagent0546"
    buf = {
        "cards": np.zeros((1, 160, 21), np.int32),
        "global": np.array(
            [
                [
                    8000,
                    8000,
                    1,
                    C.PHASE_MAIN1,
                    1,
                    1,
                    38,
                    5,
                    0,
                    0,
                    0,
                    0,
                    14,
                    41,
                    5,
                    0,
                    0,
                    0,
                    0,
                    15,
                    0,
                    C.MSG_SELECT_IDLECMD,
                    0,
                    1,
                ]
            ],
            np.int32,
        ),
        "actions": np.zeros((1, 64, 10), np.int32),
        "action_mask": np.zeros((1, 64), bool),
        "history": np.zeros((1, 32, 13), np.int32),
        "history_mask": np.zeros((1, 32), bool),
        "player": np.array([0], np.int32),
        "game": np.array([7], np.int64),
    }
    buf["cards"][0, 0] = card_row()
    for i, kind in enumerate((Kind.SUMMON, Kind.SPSUMMON, Kind.TO_END)):
        buf["actions"][0, i] = opt(kind, row=0)
    buf["action_mask"][0, :3] = True
    out = pol(buf, np.array([0]))
    assert out.shape == (1,) and out.dtype == np.int32 and 0 <= int(out[0]) < 3
    assert (7, 0) in pol._states  # LSTM state kept per (game, seat)
    out2 = pol(buf, np.array([0]))
    assert 0 <= int(out2[0]) < 3  # second call reuses the carried state without error
