"""Builder for the static annotation viewer: compact games, run history, exit codes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("build_viewer", ROOT / "scripts" / "build_viewer.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_viewer"] = mod
    spec.loader.exec_module(mod)
    return mod


BV = _load()

COUNTS0 = [35, 5, 0, 0, 0, 0, 15]
COUNTS1 = [35, 5, 0, 0, 0, 0, 14]


def _turn(tp, adv=0, v=None):
    t = {"turn_player": tp, "lp": [8000, 8000], "counts": {"0": list(COUNTS0), "1": list(COUNTS1)},
         "card_adv": adv, "msg": 16}
    if v is not None:
        t.update(v)
    return t


def _eval(k, turn, label="none", regret_pp=0.0):
    return {"k": k, "turn": turn, "seat": 0, "kind": "own", "label": label,
            "regret": regret_pp / 50, "regret_pp": regret_pp, "se": 0.01, "gap_se": 1.0,
            "significant": label != "none", "played": 1, "chosen": 1, "best": 0,
            "overridden": False, "v_best": 0.2, "phat_searcher": 0.6, "v0": 0.2, "p0": 0.6,
            "played_desc": "plays A", "best_desc": "plays B", "msg": 11, "n_worlds": 4,
            "cands": [{"opt": 0, "mean": 0.2, "desc": "plays B", "win_pct": 60.0, "se_mean": 0.02},
                      {"opt": 1, "mean": 0.1, "desc": "plays A", "win_pct": 55.0, "se_mean": 0.02}]}


def _ann(idx, with_values):
    turns = {"1": _turn(0), "2": _turn(1, adv=2), "3": _turn(0, adv=-1)}
    ann = {"game_idx": idx, "decks": ["TenpaiDragon", "SkyStrikerAce"], "seed": 100 + idx,
           "seat": 0, "s_first": True, "deck_s": "TenpaiDragon", "winner": 0,
           "truncated": False, "n_choices": 10, "turns": turns,
           "evals": [_eval(2, 2), _eval(5, 3, label="mistake", regret_pp=12.345678)],
           "curve": {"1": None, "2": 0.123456, "3": 0.4},
           "tempo": {"2": {"to_turn": 3, "dp_tp": 0.123456, "dp_tp_pp": 12.3456,
                           "turn_player": 1, "tempo_cards": 1, "tempo_lp": 0}},
           "swing_turn": 2,
           "summary": {"n_searched": 2, "n_overridden": 0,
                       "labels": {"inaccuracy": 0, "mistake": 1, "blunder": 0},
                       "mean_regret_pp": 6.172839}}
    if with_values:
        ann["curve_full"] = [{"k": k, "actor": k % 2, "turn": 1 + k // 4,
                              "v": 0.111111, "v0": 0.222222} for k in range(10)]
        for t, v in (("1", (0.1, None)), ("2", (0.2, -0.05)), ("3", (None, -0.3))):
            turns[t]["v_p0"], turns[t]["v_p1"] = v
            turns[t]["v_comb"] = 0.123456
            turns[t]["disagree"] = 0.05
        ann["tempo_search"] = ann["tempo"]
        ann["swing_turn_search"] = ann["swing_turn"]
        ann["tempo"] = {"1": {"to_turn": 2, "dp_tp": 0.05, "dp_tp_pp": 5.0,
                              "turn_player": 0, "tempo_cards": 0, "tempo_lp": 0}}
        ann["swing_turn"] = 1
    return ann


def _summary():
    return {"meta": {"input_dirs": ["x"], "n_games": 2, "n_truncated": 0,
                     "n_searched_decisions": 4, "commit": "abc",
                     "timestamp": "2026-10-16T00:00:00+00:00"},
            "calibration": {"bins": [{"lo": 0.0, "hi": 0.1, "n": 1, "won": 0.0, "mean_p": 0.05}],
                            "brier_by_bucket": {"1-2": 0.123456}, "ece": 0.0123456},
            "sign_accuracy": {"search_eval": {"<=2": {"n": 1, "acc": 0.75}},
                              "card_adv": {"<=2": {"n": 1, "acc": 0.5}},
                              "lp_lead": {"<=2": {"n": 0, "acc": None}}},
            "mining": [{"kind": "own", "msg": 11, "bucket": "1-2", "seat_first": True,
                        "deck_s": "TenpaiDragon", "opportunities": 2, "hits": 1, "rate": 0.5}],
            "tempo": {"by_turn_seat": [{"turn": 2, "seat": 1, "n": 1, "mean_pp": 12.3,
                                        "p10_pp": 12.3, "p90_pp": 12.3}],
                      "swing_share": {"1": 0.5, "2": 0.5, "3": 0.0, "4+": 0.0},
                      "swing_winner_winrate": 1.0, "n_swing_games": 2}}


def _events_line(idx):
    return {"game_idx": idx, "seed": 100 + idx, "decks": ["TenpaiDragon", "SkyStrikerAce"],
            "winner": 0, "final_lp": [8000, 0],
            "turns": [{"turn": t, "turn_player": (t + 1) % 2,
                      "players": [{"activations": 1, "activations_opp_turn": 0,
                                   "negated_by_p_suffered": 0, "damage_taken": 0,
                                   "lp_paid": 0, "lp_recovered": 0, "normal_summons": 1,
                                   "special_summons": 0, "attacks_declared": 0, "draws": 5},
                                  {"activations": 2, "activations_opp_turn": 1,
                                   "negated_by_p_suffered": 1, "damage_taken": 0,
                                   "lp_paid": 0, "lp_recovered": 0, "normal_summons": 0,
                                   "special_summons": 0, "attacks_declared": 0, "draws": 5}]}
                     for t in (1, 2, 3)]}


def _write_ann(tmp: Path, name="ann"):
    d = tmp / name
    (d / "games").mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(_summary()))
    (d / "games" / "game-0.json").write_text(json.dumps(_ann(0, False)))
    (d / "games" / "game-1.json").write_text(json.dumps(_ann(1, True)))
    ev = tmp / "events.jsonl"
    ev.write_text("\n".join(json.dumps(_events_line(i)) for i in (0, 1)) + "\n")
    return d, ev


def test_build_appends_and_compacts(tmp_path):
    ann, ev = _write_ann(tmp_path)
    out = tmp_path / "viewer"
    assert BV.main(["--run", f"r1={ann},events={ev}", "--out", str(out)]) == 0
    runs = json.loads((out / "data" / "runs.json").read_text())
    assert [r["name"] for r in runs] == ["r1"]
    r1 = runs[0]
    assert r1["n_games"] == 2 and r1["date"] == "2026-10-16T00:00:00+00:00"
    assert r1["interruptions_per_game"] == pytest.approx(3.0)
    assert r1["negations_per_game"] == pytest.approx(3.0)
    assert r1["labels_per_game"]["mistake"] == pytest.approx(1.0)

    g0 = json.loads((out / "data" / "r1" / "0.json").read_text())
    assert g0["decks"] == ["TenpaiDragon", "SkyStrikerAce"]
    assert g0["curve_full"] is None and g0["tempo_search_pp"] == {}
    assert g0["turns"][0]["ev"][1]["activations_opp_turn"] == 1
    # floats rounded to 3 decimals
    assert g0["evals"][1]["regret_pp"] == pytest.approx(12.346)
    assert g0["curve"]["2"] == pytest.approx(0.123)
    g1 = json.loads((out / "data" / "r1" / "1.json").read_text())
    assert len(g1["curve_full"]) == 10
    assert g1["tempo_search_pp"] == {"2": pytest.approx(12.346)}
    assert g1["turns"][1]["v_comb"] == pytest.approx(0.123)
    index = json.loads((out / "data" / "r1" / "index.json").read_text())
    assert [r["idx"] for r in index] == [0, 1]
    assert set(index[0]) == {"idx", "decks", "seats", "winner", "turns", "n_labels",
                             "mean_regret", "swing_turn", "final_card_adv"}
    assert index[0]["n_labels"] == 1
    assert (out / "data" / "r1" / "summary.json").is_file()
    assert (out / "index.html").is_file()

    # second run keeps the first (history); re-running r1 updates it in place
    ann2, _ = _write_ann(tmp_path, name="ann2")
    assert BV.main(["--run", f"r2={ann2}", "--out", str(out), "--max-games-per-run", "1"]) == 0
    runs = json.loads((out / "data" / "runs.json").read_text())
    assert [r["name"] for r in runs] == ["r1", "r2"]
    assert runs[1]["n_games"] == 1 and runs[1]["interruptions_per_game"] is None
    assert BV.main(["--run", f"r1={ann2}", "--out", str(out)]) == 0
    runs = json.loads((out / "data" / "runs.json").read_text())
    assert [r["name"] for r in runs] == ["r1", "r2"]
    assert runs[0]["n_games"] == 2  # updated, not duplicated


def test_exit_codes(tmp_path):
    out = tmp_path / "viewer"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as e:
        BV.main(["--run", f"bad={empty}", "--out", str(out)])
    assert e.value.code != 0  # missing summary.json
    (empty / "summary.json").write_text(json.dumps(_summary()))
    with pytest.raises(SystemExit) as e:
        BV.main(["--run", f"bad={empty}", "--out", str(out)])
    assert e.value.code != 0  # 0 games


def _aux_turn():
    t = _turn(0)
    t["aux"] = {
        "0": {"k": 0, "pred_lp": [100, -200], "pred_cards": [0.0] * 14,
              "pred_ends": 0.2, "pred_turns": 4.0,
              "real_lp": [-500, -1000], "real_cards": [0.0] * 14,
              "real_ends": 0, "real_turns": 2},
        "1": {"k": 1, "pred_lp": [0, 0], "pred_cards": [0.0] * 14,
              "pred_ends": 0.8, "pred_turns": 3.0,
              "real_lp": None, "real_cards": None, "real_ends": 1, "real_turns": 2},
    }
    return t


def _aux_summary():
    s = _summary()
    s["aux"] = {
        "n_points": 2,
        "ends": {"n": 2, "brier": 0.1, "brier_base": 0.25, "ece": 0.05,
                 "base_rate": 0.5, "bins": [{"lo": 0.0, "hi": 0.1, "n": 1, "rate": 0.0, "mean_p": 0.05}]},
        "d_lp": {"n": 2, "mae": 100.0, "mse": 10000.0, "mse_base": 20000.0, "r2": 0.5},
        "d_cards": {"n": 14, "mae": 0.1, "mse": 0.01, "mse_base": 0.02, "r2": 0.5},
        "turns_left": {"n": 2, "mae": 0.5, "mse": 0.25, "mse_base": 1.0, "r2": 0.75, "mean": 2.0},
    }
    return s


def test_build_with_aux(tmp_path):
    d = tmp_path / "ann-aux"
    (d / "games").mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(_aux_summary()))
    ann = _ann(0, True)
    ann["turns"]["1"] = _aux_turn()
    (d / "games" / "game-0.json").write_text(json.dumps(ann))
    out = tmp_path / "viewer-aux"
    assert BV.main(["--run", f"ra={d}", "--out", str(out)]) == 0
    g0 = json.loads((out / "data" / "ra" / "0.json").read_text())
    assert "aux" in g0["turns"][0]
    assert g0["turns"][0]["aux"]["0"]["pred_ends"] == pytest.approx(0.2)
    runs = json.loads((out / "data" / "runs.json").read_text())
    assert runs[0]["aux"]["ends_brier"] == pytest.approx(0.1)
    assert runs[0]["aux"]["d_lp_mae"] == pytest.approx(100.0)
    # run without aux still builds and has no aux headline
    ann2, _ = _write_ann(tmp_path, name="ann-noaux")
    assert BV.main(["--run", "rb=" + str(ann2), "--out", str(out)]) == 0
    runs = json.loads((out / "data" / "runs.json").read_text())
    by_name = {r["name"]: r for r in runs}
    assert "aux" not in by_name["rb"]
