"""Unit tests for analysis/annotate.py: pure functions, no engine, no torch."""

import math

import pytest

from ygo_zero.analysis import annotate as A

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
