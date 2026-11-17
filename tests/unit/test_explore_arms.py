"""Pure parts of scripts/audit/explore_arms.py: the window filter and the merge math. No engine."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parents[2] / "scripts" / "audit"

sys.path.insert(0, str(AUDIT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_mod", AUDIT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _obs(g5: int = 0, g20: int = 0, kinds=(6,), mask=None):
    g = np.zeros(21, np.int32)
    g[5], g[20] = g5, g20
    actions = np.zeros((len(kinds), 10), np.int32)
    actions[:, 0] = kinds
    m = np.ones(len(kinds), bool) if mask is None else np.asarray(mask, bool)
    return {"global": g, "actions": actions, "action_mask": m}


def test_filter_keeps_opponent_chain0_activation():
    ea = _load("explore_arms")
    assert ea.filter_window(_obs()) == (True, "")


def test_filter_reasons():
    ea = _load("explore_arms")
    assert ea.filter_window(_obs(g5=1)) == (False, "own_turn")
    assert ea.filter_window(_obs(g20=2)) == (False, "chain")
    assert ea.filter_window(_obs(kinds=(11,))) == (False, "no_activate")
    assert ea.filter_window(_obs(kinds=(6,), mask=(False,))) == (False, "no_activate")
    assert ea.filter_window(_obs(kinds=(6, 11, 6), mask=(True, True, False))) == (True, "")


def test_activate_indices_uses_mask():
    ea = _load("explore_arms")
    assert ea.activate_indices(_obs(kinds=(11, 6, 6), mask=(True, True, False))) == [1]


def _draw(won, field, index, code=100, card="A"):
    return {"won": won, "opp_field_end_turn": field, "index": index, "code": code, "card": card,
            "label": f"activate {card}"}


def _arm(won, field):
    return {"won": won, "opp_field_end_turn": field}


def _variant(p, s, rand):
    return {"p": _arm(*p), "search": _arm(*s), "rand": rand}


def _rows():
    r1 = {"search_index": 5, "variants": [
        _variant((0, 2), (1, 1), [_draw(1, 1, 5), _draw(1, 1, 5)]),
        _variant((1, 2), (1, 2), [_draw(0, 3, 5), _draw(0, 3, 5)])]}
    r2 = {"search_index": 9, "variants": [
        _variant((0, 3), (0, 3), [_draw(1, 1, 7, 200, "B"), _draw(0, 4, 8, 200, "B")])]}
    r3 = {"search_index": 5, "variants": [
        _variant((1, 2), (0, 4), [_draw(0, 4, 5)])]}
    return [r1, r2, r3]


def test_summarise_merge_math():
    ea = _load("explore_arms")
    s = ea.summarise(_rows(), jobs=3, n_boot=200, seed=0)
    assert s["valid_windows"] == 3 and s["rows"] == 3
    assert s["mean_win"]["p"] == 0.5
    assert s["mean_win"]["search"] == round(1 / 3, 4)
    assert s["mean_win"]["rand"] == round(1 / 3, 4)
    for arm in ("search_minus_p", "rand_minus_p"):
        d = s["mean_win"][arm]
        assert d["mean"] == round(-1 / 6, 4)
        assert d["better"] == 1 and d["worse"] == 1 and d["sign_p"] == 0.75
        assert d["ci95"][0] <= d["mean"] <= d["ci95"][1]
    assert s["opp_field_end_turn"]["p"] == round(7 / 3, 3)
    for arm in ("search_minus_p", "rand_minus_p"):
        d = s["opp_field_end_turn"][arm]
        assert d["mean"] == 0.5 and d["better"] == 1 and d["worse"] == 1
        assert d["dir"] == "lower_better"


def test_summarise_per_card_and_search_share():
    ea = _load("explore_arms")
    s = ea.summarise(_rows(), jobs=3, n_boot=50)
    assert [(r["code"], r["n_draws"], r["mean_won_diff"]) for r in s["per_card"]] == [
        (100, 5, -0.2), (200, 2, 0.5)]
    assert s["rand_equals_search"] == {"draws": 7, "equal": 5, "share": round(5 / 7, 4)}


def test_summarise_empty_and_bootstrap_deterministic():
    ea = _load("explore_arms")
    s = ea.summarise([], jobs=0)
    assert s["valid_windows"] == 0 and s["per_card"] == []
    assert s["rand_equals_search"]["share"] is None
    a = ea.bootstrap_ci([1, 2, 3, 4], 200, 7)
    assert a == ea.bootstrap_ci([1, 2, 3, 4], 200, 7)
    assert a[0] <= 2.5 <= a[1]
    assert ea.sign_p(1, 1) == 0.75 and ea.sign_p(0, 0) == 1.0
