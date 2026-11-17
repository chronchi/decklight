"""Pure summary math of scripts/audit/public_policy_compare.py (no model, no engine)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "audit"))
import public_policy_compare as PPC

ARMS = ["X", "OC"]


def _row(tv, xp, ocp, ours=True, turn=1, going_first=True,
         message="SelectIdleCmd", xarg="a", ocarg="a", gf=False):
    probs = {"X": list(xp), "OC": list(ocp)}
    argmax = {"X": xarg, "OC": ocarg}
    return {
        "stream": "runs/x/g1/s.stream.jsonl.gz", "game": "runs/x/g1", "input_set": "pat",
        "duel": 1, "turn": turn, "ours": ours, "going_first": going_first, "message": message,
        "options": ["a", "b", "to_end"], "played": 0, "played_label": "a",
        "probs": probs, "argmax": argmax, "argmax_index": {"X": 0, "OC": 0},
        "p_played": {"X": xp[0], "OC": ocp[0]}, "tv": tv, "agree": xarg == ocarg,
        "first_gf_t1_main1": gf, "hand": ["a", "b"], "to_end_index": 2,
    }


def test_summarize_groups_and_means():
    rows = [
        _row(0.0, [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]),
        _row(
            0.4, [0.6, 0.2, 0.2], [0.2, 0.4, 0.4], ours=False, turn=2,
            going_first=False, message="SelectChain", xarg="a", ocarg="b",
        ),
    ]
    s = PPC.summarize(rows, ARMS)
    assert s["groups"]["all"]["decisions"] == 2
    assert s["groups"]["our_turn"]["decisions"] == 1
    assert s["groups"]["opp_turn"]["decisions"] == 1
    assert s["groups"]["gf_t1"]["decisions"] == 1
    assert s["groups"]["all"]["agreement"] == 0.5
    assert s["groups"]["all"]["tv_mean"] == 0.2
    assert s["groups"]["all"]["p_played_X"] == 0.7
    assert s["groups"]["all"]["p_played_OC"] == 0.5
    assert s["groups"]["all"]["agree_X|OC"] == 0.5
    assert [e["tv"] for e in s["top_tv"]] == [0.4, 0.0]


def test_summarize_gf_main1_reports_to_end_per_arm():
    rows = [_row(0.0, [0.5, 0.1, 0.4], [0.3, 0.1, 0.6], gf=True)]
    s = PPC.summarize(rows, ARMS)
    (e,) = s["gf_t1_main1"]
    assert e["X_p_to_end"] == 0.4
    assert e["OC_p_to_end"] == 0.6
    assert e["X_argmax"] == "a" and e["OC_argmax"] == "a"
    assert e["hand"] == ["a", "b"]
    assert e["to_end_option"] == "to_end"


def test_summarize_empty_groups():
    s = PPC.summarize([], ARMS)
    assert s["groups"]["all"] == {"decisions": 0}
    assert s["top_tv"] == [] and s["gf_t1_main1"] == []


def test_tv_and_pct():
    assert PPC._tv([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert PPC._tv([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert PPC._pct([], 0.9) == 0.0
    assert PPC._pct([1.0, 2.0, 3.0], 0.0) == 1.0
    assert PPC._pct([1.0, 2.0, 3.0], 1.0) == 3.0
