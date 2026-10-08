"""Paired ygo-agent gate: exact sign-test p-value, pairing coverage, exit 2 on mismatched schedules."""

import importlib.util
import math
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "scripts" / "audit" / "paired_gate.py"


def _gate():
    spec = importlib.util.spec_from_file_location("paired_gate_mod", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(da, db, seed, first, win):
    return [da, db, seed, first, win, 10, False]


def _report(path: Path, wins: list[float]):
    decks = ["Alpha", "Beta"]
    rows = [_row(decks[0], decks[1], i // 2, i % 2 == 0, w) for i, w in enumerate(wins)]
    path.write_text(__import__("json").dumps({"matches": {"ygoagent": {"per_game": rows}}}))


def test_signtest_p_hand_computed():
    mod = _gate()
    assert mod.signtest_p(0, 0) == 1.0
    # 7 strict wins for B, 1 for A (n=8): two-sided p = 2*(C(8,0)+C(8,1))/256 = 18/256
    assert mod.signtest_p(1, 7) == 18 / 256 == 0.0703125
    assert mod.signtest_p.__doc__ and mod.diff_ci([(1.0 - 0.0)] * 4)[0] == 1.0


def test_pairs_counts_diff_and_ci(tmp_path, capsys):
    mod = _gate()
    # 8 paired deals: A goes 1,1,1,1,1,1,1,0 vs B 0,0,0,0,0,0,0,1 -> W->L 7, L->W 1, diff +0.75
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _report(a, [1.0] * 7 + [0.0])
    _report(b, [0.0] * 7 + [1.0])
    assert mod.main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "pairs 8" in out and "W->L    7" in out and "L->W    1" in out
    assert "p=0.070312" in out
    assert "diff +0.750" in out
    # normal-approx CI on per-pair diffs {+1 x7, -1}: mean .75, sd .4629.., h = 1.96*sd/sqrt(8)
    diffs = [1.0] * 7 + [-1.0]
    mean = sum(diffs) / 8
    var = sum((d - mean) ** 2 for d in diffs) / 7
    h = 1.96 * math.sqrt(var / 8)
    assert f"[{mean - h:+.3f}, {mean + h:+.3f}]" in out


def test_mismatched_schedules_exit_2(tmp_path, capsys):
    mod = _gate()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _report(a, [1.0] * 8)
    rows = [_row("Alpha", "Beta", s, f, 0.0) for s in range(2) for f in (True, False)]
    rows += [_row("Gamma", "Beta", s, True, 0.0) for s in range(4)]  # half the keys differ
    b.write_text(__import__("json").dumps({"matches": {"ygoagent": {"per_game": rows}}}))
    assert mod.main([str(a), str(b)]) == 2
    assert "fewer than 90%" in capsys.readouterr().out


def test_missing_match_exit_2(tmp_path, capsys):
    mod = _gate()
    a = tmp_path / "a.json"
    a.write_text("{}")
    b = tmp_path / "b.json"
    _report(b, [1.0] * 4)
    assert mod.main([str(a), str(b)]) == 2
