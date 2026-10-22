"""Unit tests for analysis/annotate.py: pure functions, no engine, no torch."""

import math

import pytest

from decklight.analysis import annotate as A

BASE = [40, 5, 0, 0, 0, 0, 15]


def snap(k, turn, actor=0, tp=0, lp0=8000, lp1=8000, c0=None, c1=None, msg=7):
    return {
        "k": k,
        "turn": turn,
        "actor": actor,
        "turn_player": tp,
        "lp": [lp0, lp1],
        "counts": {0: list(BASE if c0 is None else c0), 1: list(BASE if c1 is None else c1)},
        "msg": msg,
    }


def entry(k, best_w, played_w, seat=0, turn=1, kind="own", chosen=1, played=0, cands=(1, 0), best=None):
    e = {
        "k": k,
        "seat": seat,
        "kind": kind,
        "turn": turn,
        "cands": list(cands),
        "samples": [[float(best_w[0]), float(best_w[1])], [float(played_w[0]), float(played_w[1])]],
        "chosen": chosen,
        "played": played,
    }
    if best is not None:
        e["best"] = best
    return e


def rec_of(search, winner=0, seat=0, turns=2, idx=0):
    return {
        "game_idx": idx,
        "decks": ["a", "b"],
        "seed": 1,
        "s_seat": seat,
        "s_first": seat == 0,
        "deck_s": "a",
        "deck_r": "b",
        "winner": winner,
        "truncated": False,
        "turns": turns,
        "search": search,
    }


def test_labels_thresholds():
    # regret 0.05 (< 0.1), significant -> still none
    ev = A.eval_entry(entry(0, (0.2, 0.2), (0.15, 0.15)))
    assert ev["regret"] == pytest.approx(0.05)
    assert ev["significant"] and ev["label"] == "none"
    # 0.12 -> inaccuracy; 0.25 -> mistake; 0.35 -> blunder (zero spread => significant)
    assert A.eval_entry(entry(0, (0.12, 0.12), (0.0, 0.0)))["label"] == "inaccuracy"
    assert A.eval_entry(entry(0, (0.25, 0.25), (0.0, 0.0)))["label"] == "mistake"
    assert A.eval_entry(entry(0, (0.35, 0.35), (0.0, 0.0)))["label"] == "blunder"


def test_significance_needs_two_se():
    # regret 0.2 but noisy worlds (gap < 2 SE) -> none
    ev = A.eval_entry(entry(0, (0.9, -0.5), (0.9, -0.9)))
    assert ev["regret"] == pytest.approx(0.2)
    assert ev["gap_se"] < 2
    assert ev["label"] == "none"


def test_identical_worlds_no_label():
    ev = A.eval_entry(entry(0, (0.4, 0.4), (0.4, 0.4)))
    assert ev["regret"] == 0.0
    assert ev["label"] == "none"


def test_played_outside_cands():
    e = entry(0, (0.5, 0.5), (0.0, 0.0), played=9)
    e["cands"] = [1, 0]
    ev = A.eval_entry(e)
    assert ev["regret"] is None and ev["se"] is None and ev["label"] == "none"


def test_explicit_best_overrides_argmax():
    e = entry(0, (0.1, 0.1), (0.9, 0.9), best=1)  # best = option 1 = lower mean
    ev = A.eval_entry(e)
    assert ev["best"] == 1
    assert ev["regret"] == pytest.approx(0.1 - 0.9)


def test_nan_raises():
    e = entry(0, (0.5, 0.5), (0.0, 0.0))
    e["samples"][0][0] = float("nan")
    with pytest.raises(ValueError):
        A.eval_entry(e)


def test_v0_sign_flips_with_seat():
    e0 = A.eval_entry(entry(0, (0.6, 0.6), (0.0, 0.0), seat=0))
    e1 = A.eval_entry(entry(0, (0.6, 0.6), (0.0, 0.0), seat=1))
    assert e0["v0"] == pytest.approx(-e1["v0"])
    assert e0["phat_searcher"] == pytest.approx((0.6 + 1) / 2)


def _two_turn_game(**kw):
    """Two searched decisions on turns 1 and 3 (curve gap at turn 2)."""
    s1 = [snap(0, 1, actor=0, tp=0), snap(1, 2, actor=1, tp=1), snap(2, 3, actor=0, tp=0)]
    search = [
        entry(0, (0.0, 0.0), (-0.4, -0.4), seat=0, turn=1),
        entry(2, (0.4, 0.4), (0.0, 0.0), seat=0, turn=3),
    ]
    r = rec_of(search, turns=3, **kw)
    return A.annotate_game(r, s1, s1[-1], {0: [(1, "A"), (0, "B")], 2: [(1, "C"), (0, "D")]})


def test_curve_gap_and_tempo_sign():
    ann = _two_turn_game()
    assert ann["curve"][1] == pytest.approx(0.0)
    assert ann["curve"][2] is None
    assert ann["curve"][3] == pytest.approx(0.4)
    # next available curve after turn 1 is turn 3; turn player 0 gains 20 win-pp
    assert ann["tempo"][1]["to_turn"] == 3
    assert ann["tempo"][1]["dp_tp_pp"] == pytest.approx(20.0)
    assert ann["swing_turn"] == 1


def test_tempo_flips_for_player_one():
    s1 = [snap(0, 1, actor=1, tp=1), snap(1, 2, actor=0, tp=0)]
    search = [
        entry(0, (0.0, 0.0), (-0.4, -0.4), seat=1, turn=1),
        entry(1, (-0.4, -0.4), (-0.4, -0.4), seat=1, turn=2),
    ]
    # seat 1: v0 = -v_best -> curve 1: 0.0, curve 2: 0.4
    r = rec_of(search, winner=1, seat=1, turns=2)
    ann = A.annotate_game(r, s1, s1[-1], {0: [(1, "A"), (0, "B")], 1: [(1, "C"), (0, "D")]})
    assert ann["curve"][1] == pytest.approx(0.0)
    assert ann["curve"][2] == pytest.approx(0.4)
    # player-0 win% rose, so the turn-1 player (seat 1) lost 20pp of tempo
    assert ann["tempo"][1]["dp_tp_pp"] == pytest.approx(-20.0)


def test_calibration_bins_and_summary():
    anns = []
    # 2 games: searcher wins with phat ~0.96 (bin 9), loses with phat ~0.45 (bin 4)
    for i, (seat, winner, v) in enumerate([(0, 0, 0.92), (0, 1, -0.1)]):
        s = [snap(0, 1)]
        search = [entry(0, (v, v), (v - 0.5, v - 0.5), seat=seat, turn=1)]
        anns.append(
            A.annotate_game(
                rec_of(search, winner=winner, seat=seat, turns=1, idx=i), s, s[-1], {0: [(1, "A"), (0, "B")]}
            )
        )
    sm = A.summarize(anns, ["d"], "abc")
    assert sm["meta"]["n_games"] == 2
    assert sm["meta"]["n_searched_decisions"] == 2
    b9, b4 = sm["calibration"]["bins"][9], sm["calibration"]["bins"][4]
    assert (b9["n"], b9["won"]) == (1, 1.0)
    assert (b4["n"], b4["won"]) == (1, 0.0)
    assert sm["calibration"]["ece"] == pytest.approx((abs(1.0 - 0.96) + abs(0.0 - 0.45)) / 2)
    assert sm["sign_accuracy"]["search_eval"]["<=2"]["n"] == 2
    assert sm["sign_accuracy"]["search_eval"]["<=2"]["acc"] == 1.0


def test_mining_and_card_adv_accuracy():
    ann = _two_turn_game()
    sm = A.summarize([ann], ["d"], "abc")
    assert sm["meta"]["n_searched_decisions"] == 2
    # entry k=0: regret 0.4 -> blunder; k=2: regret 0.4 -> blunder
    assert ann["summary"]["labels"]["blunder"] == 2
    assert len(sm["mining"]) == 2  # turns 1 and 3 are different buckets
    assert all(r["hits"] == 1 and r["opportunities"] == 1 for r in sm["mining"])


def test_markdown_renders():
    ann = _two_turn_game()
    md = A.render_game_md(ann)
    assert "blunder" in md and "Turn 1" in md
    sm = A.summarize([ann], ["d"], "abc")
    smd = A.render_summary_md(sm)
    assert "Calibration" in smd and "Tempo" in smd
    # no NaN sneaks into JSON-serializable output
    import json

    json.dumps(ann)
    json.dumps(sm)
    assert math.isfinite(ann["summary"]["mean_regret_pp"])


EXPECTED_NO_VALUE_KEYS = {
    "game_idx", "decks", "seed", "seat", "s_first", "deck_s", "winner", "truncated",
    "n_choices", "turns", "evals", "curve", "tempo", "swing_turn", "summary",
}


def test_no_values_output_shape_unchanged():
    ann = _two_turn_game()
    assert set(ann.keys()) == EXPECTED_NO_VALUE_KEYS
    assert "curve_full" not in ann and "tempo_search" not in ann
    assert all("v_comb" not in ts for ts in ann["turns"].values())


def _values_of(snaps, fn):
    """A score_decisions-style values line: v = fn(turn, actor)."""
    return {
        "game_idx": 0,
        "seed": 1,
        "decks": ["a", "b"],
        "s_seat": 0,
        "k": [s["k"] for s in snaps],
        "actor": [s["actor"] for s in snaps],
        "turn": [s["turn"] for s in snaps],
        "v": [round(fn(s["turn"], s["actor"]), 4) for s in snaps],
    }


def _valued_game():
    """Three turns, both seats act every turn; v = 0.1 * turn * (+1 actor 0, -1 actor 1)."""
    s1 = [
        snap(0, 1, actor=0, tp=0),
        snap(1, 1, actor=1, tp=0),
        snap(2, 2, actor=1, tp=1),
        snap(3, 2, actor=0, tp=1),
        snap(4, 3, actor=0, tp=0),
        snap(5, 3, actor=1, tp=0),
    ]
    search = [entry(0, (0.0, 0.0), (-0.4, -0.4), seat=0, turn=1)]
    r = rec_of(search, turns=3)
    values = _values_of(s1, lambda t, a: 0.1 * t * (1 if a == 0 else -1))
    plain = A.annotate_game(r, s1, s1[-1], {0: [(1, "A"), (0, "B")]})
    valued = A.annotate_game(r, s1, s1[-1], {0: [(1, "A"), (0, "B")]}, values)
    return plain, valued


def test_values_curve_full_and_v_comb():
    plain, valued = _valued_game()
    assert "curve_full" not in plain
    cf = valued["curve_full"]
    assert len(cf) == 6
    # v0 is player 0's view: v if actor 0 else -v -> 0.1 * turn everywhere here
    for e in cf:
        assert e["v0"] == pytest.approx(0.1 * e["turn"])
    for t in (1, 2, 3):
        ts = valued["turns"][t]
        assert ts["v_p0"] == pytest.approx(0.1 * t)
        assert ts["v_p1"] == pytest.approx(-0.1 * t)
        assert ts["v_comb"] == pytest.approx(0.1 * t)
        assert ts["disagree"] == pytest.approx(0.0)


def test_values_tempo_from_full_curve_and_search_kept():
    plain, valued = _valued_game()
    # search tempo preserved under its new name
    assert valued["tempo_search"] == plain["tempo"]
    assert valued["swing_turn_search"] == plain["swing_turn"]
    # full-curve tempo: v_comb 0.1 -> 0.2 -> 0.3, i.e. +5 win-pp per turn for player 0's view
    assert set(valued["tempo"]) == {1, 2}
    assert valued["tempo"][1]["to_turn"] == 2
    assert valued["tempo"][1]["dp_tp_pp"] == pytest.approx(5.0)  # turn 1 player is seat 0
    assert valued["tempo"][2]["dp_tp_pp"] == pytest.approx(-5.0)  # turn 2 player is seat 1
    assert valued["swing_turn"] in (1, 2)


def test_values_summary_sections():
    _, valued = _valued_game()
    sm = A.summarize([valued], ["d"], "abc")
    assert "value_calibration" in sm and "v_comb_sign" in sm and "disagree" in sm
    assert sm["v_comb_sign"]["<=2"]["n"] == 2  # turns 1 and 2 share the bucket
    assert sm["v_comb_sign"]["<=2"]["acc"] == 1.0  # winner is seat 0, v_comb > 0
    assert sm["disagree"]["<=2"]["mean_abs"] == pytest.approx(0.0)
    smd = A.render_summary_md(sm)
    assert "v_comb" in smd and "Disagreement" in smd
    md = A.render_game_md(valued)
    assert "v_comb" in md  # eval column now comes from the full curve
    import json

    json.dumps(valued)
    json.dumps(sm)


def test_values_single_seat_turn_fallback():
    # a turn where only seat 1 acts: v_comb is seat 1's value flipped to player 0's view
    s1 = [snap(0, 1, actor=1, tp=1)]
    r = rec_of([], turns=1)
    values = _values_of(s1, lambda t, a: -0.2)
    ann = A.annotate_game(r, s1, s1[-1], {}, values)
    assert ann["turns"][1]["v_p0"] is None
    assert ann["turns"][1]["v_comb"] == pytest.approx(0.2)
    assert ann["turns"][1]["disagree"] is None


def test_values_length_mismatch_raises():
    s1 = [snap(0, 1), snap(1, 1)]
    r = rec_of([], turns=1)
    bad = {"k": [0], "actor": [0], "turn": [1], "v": [0.1]}
    with pytest.raises(ValueError):
        A.annotate_game(r, s1, s1[-1], {}, bad)


def _aux_values_of(snaps, d_lp=None, d_cards=None, ends=None, turns_left=None):
    n = len(snaps)
    base = _values_of(snaps, lambda t, a: 0.1)
    base["aux"] = {
        "d_lp": [[0, 0] for _ in range(n)] if d_lp is None else d_lp,
        "d_cards": [[0.0] * 14 for _ in range(n)] if d_cards is None else d_cards,
        "ends": [0.1 for _ in range(n)] if ends is None else ends,
        "turns_left": [2.0 for _ in range(n)] if turns_left is None else turns_left,
    }
    return base


def _aux_game():
    """Turn 1 (P0 to move, both seats act), turn 2 (P1 to move, both act), turn 3 (P0)."""
    c_hand2 = list(BASE)
    c_hand2[1] = 4  # one fewer hand card
    s1 = [
        snap(0, 1, actor=0, tp=0, lp0=8000, lp1=8000),
        snap(1, 1, actor=1, tp=0, lp0=8000, lp1=8000),
        snap(2, 2, actor=1, tp=1, lp0=7500, lp1=8000, c0=c_hand2),
        snap(3, 2, actor=0, tp=1, lp0=7500, lp1=8000, c0=c_hand2),
        snap(4, 3, actor=0, tp=0, lp0=7500, lp1=7000),
    ]
    r = rec_of([], turns=3)
    values = _aux_values_of(
        s1,
        d_lp=[[100, -200], [0, -500], [0, 0], [-100, 0], [50, 50]],
        ends=[0.2, 0.8, 0.9, 0.1, 0.95],
        turns_left=[4.0, 3.0, 2.0, 1.0, 0.5],
    )
    ann = A.annotate_game(r, s1, s1[-1], {}, values)
    return ann


def test_aux_predicted_vs_realized():
    ann = _aux_game()
    # seat 0 turn 1: next own start is k4 (turn 3): me 7500-8000=-500, opp 7000-8000=-1000
    a00 = ann["turns"][1]["aux"]["0"]
    assert a00["pred_lp"] == [100, -200]
    assert a00["real_lp"] == [-500, -1000]
    assert a00["real_ends"] == 0
    assert a00["real_turns"] == 2  # end turn 3 - turn 1
    assert a00["pred_ends"] == pytest.approx(0.2)
    # seat 1 turn 1 (k1): next own start k2 (turn 2)
    a01 = ann["turns"][1]["aux"]["1"]
    assert a01["real_ends"] == 0
    assert a01["k"] == 1
    # seat 1 turn 2 (k2): no later own turn -> ends, no card/LP target
    a12 = ann["turns"][2]["aux"]["1"]
    assert a12["real_ends"] == 1
    assert a12["real_lp"] is None and a12["real_cards"] is None
    # seat 0 turn 2 (k3): next own start k4
    assert ann["turns"][2]["aux"]["0"]["real_ends"] == 0
    # turn 3 seat 0: last turn -> ends
    assert ann["turns"][3]["aux"]["0"]["real_ends"] == 1


def test_aux_summary_and_markdown():
    ann = _aux_game()
    sm = A.summarize([ann], ["d"], "abc")
    assert "aux" in sm
    assert sm["aux"]["n_points"] == 5  # (t1:2 seats) + (t2:2) + (t3:1)
    assert sm["aux"]["ends"]["n"] == 5
    assert 0.0 <= sm["aux"]["ends"]["brier"] <= 1.0
    assert sm["aux"]["d_lp"]["n"] == 6  # 3 has_next points x 2
    assert sm["aux"]["d_cards"]["n"] == 3 * 14
    assert sm["aux"]["turns_left"]["n"] == 5
    md = A.render_game_md(ann)
    assert "P(end)" in md and "T left" in md
    smd = A.render_summary_md(sm)
    assert "Aux heads" in smd
    import json

    json.dumps(ann)
    json.dumps(sm)


def test_aux_length_mismatch_raises():
    s1 = [snap(0, 1), snap(1, 1)]
    r = rec_of([], turns=1)
    values = _aux_values_of(s1)
    values["aux"]["ends"] = [0.1]  # short
    with pytest.raises(ValueError):
        A.annotate_game(r, s1, s1[-1], {}, values)
