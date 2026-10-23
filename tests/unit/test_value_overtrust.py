"""Value over-trust audit (scripts/audit/value_overtrust.py): aggregation on fake rows.

Fast: no engine, no networks, no real games.
"""

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "audit"


def _load():
    p = SCRIPTS / "value_overtrust.py"
    spec = importlib.util.spec_from_file_location("value_overtrust_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(game, p_search, p_raw, y, overridden):
    return {"game": game, "k": 0, "turn": 2, "kind": "own", "overridden": overridden,
            "chosen": 1, "played": 0, "cands": [], "p_search": p_search, "p_raw": p_raw,
            "y": y, "decks": ["a", "b"], "seed": 0, "s_seat": 0}


def test_to_p_mapping():
    vo = _load()
    assert vo.to_p(1.0) == 1.0
    assert vo.to_p(-1.0) == 0.0
    assert vo.to_p(0.0) == 0.5
    assert vo.to_p(2.0) == 1.0
    assert vo.to_p(-2.0) == 0.0


def test_group_metrics_fake():
    vo = _load()
    rows = [{"p": 1.0, "y": 1.0}, {"p": 0.5, "y": 0.0}]
    m = vo.group_metrics([{"x": 0, **r} for r in rows], "p")
    assert m["n"] == 2
    assert m["brier"] == (0.0 + 0.25) / 2
    assert m["bias"] == (0.0 + 0.5) / 2
    assert vo.group_metrics([], "p") == {"n": 0, "brier": 0.0, "bias": 0.0}


def test_summarize_groups_fake():
    vo = _load()
    rows = [
        _row(0, 0.9, 0.6, 1.0, True),   # overridden, both scored
        _row(0, 0.8, None, 0.0, True),  # overridden, chosen-only (terminal raw)
        _row(1, 0.7, 0.7, 1.0, False),  # agreeing
    ]
    s = vo.summarize(rows)
    assert (s["n_games"], s["n_overridden"], s["n_agreeing"]) == (2, 2, 1)
    assert s["chosen_overridden"]["n"] == 2
    assert s["raw_overridden"]["n"] == 1  # the None raw is excluded
    assert s["agreeing"]["n"] == 1
    # chosen bias over the two overridden: ((0.9-1) + (0.8-0)) / 2 = 0.35
    assert abs(s["chosen_overridden"]["bias"] - 0.35) < 1e-9


def test_bootstrap_cis_shape():
    vo = _load()
    rows = [_row(g, 0.6 + 0.1 * (g % 2), 0.5, float(g % 2), bool(g % 2)) for g in range(6)]
    cis = vo.bootstrap_cis(rows, 20, 0)
    assert set(cis) == {"chosen_overridden", "raw_overridden", "agreeing"}
    for group in cis.values():
        for lo, hi in group.values():
            assert lo <= hi
    gap = vo.bootstrap_gap_ci(rows, 20, 0)
    assert gap[0] is not None and gap[0] <= gap[1]
    assert vo.bootstrap_gap_ci([r for r in rows if not r["overridden"]], 20, 0) == (None, None)


def test_overtrust_verdict_fake():
    vo = _load()
    summ = {"chosen_overridden": {"bias": 0.2}}
    assert vo.overtrust_verdict(summ, {"bias_gap": (0.05, 0.3),
                                       "chosen_overridden": {"bias": (0.1, 0.3)}}).startswith("OVER-TRUST")
    assert vo.overtrust_verdict(summ, {"bias_gap": (-0.1, 0.3),
                                       "chosen_overridden": {"bias": (0.1, 0.3)}}).startswith("no significant")
    assert vo.overtrust_verdict(summ, {"bias_gap": (0.05, 0.3),
                                       "chosen_overridden": {"bias": (-0.1, 0.3)}}).startswith("no significant")
    assert vo.overtrust_verdict(summ, {}) == "no verdict (no overridden decisions)"
