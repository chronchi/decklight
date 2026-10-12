"""Paired deck A/B harness (scripts/deck_ab.py): power calc, identical-deck noise check, report shape."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "decks"


def _mod():
    spec = importlib.util.spec_from_file_location("deck_ab_mod", ROOT / "scripts" / "deck_ab.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["deck_ab_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_ydk(path: Path, main: list[int], extra: list[int]) -> None:
    path.write_text("#main\n" + "\n".join(map(str, main)) + "\n#extra\n"
                    + "\n".join(map(str, extra)) + "\n!side\n")


def test_power_numbers():
    mod = _mod()
    pc = mod.power(400, 0.75, 0.03)
    assert pc["mde"] == pytest.approx(0.105, abs=1e-9)
    assert pc["n_needed"] == 4900


def _run(mod, a: Path, b: Path, out: Path, extra_args: list[str] | None = None) -> int:
    argv = ["--deck-a", str(a), "--deck-b", str(b), "--pilot", "greedy", "--opp", "greedy",
            "--opp-decks", str(FIX), "--n-opp-decks", "2", "--seed", "0", "--seeds", "1",
            "--threads", "2", "--num-envs", "4", "--out", str(out)]
    return mod.main(argv + (extra_args or []))


def test_identical_decks_zero_diff(tmp_path, capsys):
    mod = _mod()
    out = tmp_path / "ab"
    rc = _run(mod, FIX / "BlueEyes.ydk", FIX / "BlueEyes.ydk", out)
    assert rc == 0, capsys.readouterr().out
    rep = json.loads((out / "report.json").read_text())
    assert rep["diff"] == 0.0
    assert rep["n_pairs"] == 2 * 1 * 2
    assert "MDE" in (out / "summary.txt").read_text()


def test_different_decks_report_shape(tmp_path, capsys):
    mod = _mod()
    out = tmp_path / "ab"
    rc = _run(mod, FIX / "BlueEyes.ydk", FIX / "Branded.ydk", out)
    assert rc == 0, capsys.readouterr().out
    rep = json.loads((out / "report.json").read_text())
    for field in ("win_rate_a", "ci_a", "win_rate_b", "ci_b", "diff", "se", "ci",
                  "a_lost_b_won", "a_won_b_lost", "ties", "half", "sign_p",
                  "by_seat", "per_opp", "per_game", "power", "n_pairs"):
        assert field in rep, field
    assert rep["n_pairs"] == 2 * 1 * 2 == len(rep["per_game"])
    assert set(rep["by_seat"]) == {"a_first", "a_second"}
    assert all(set(r) == {"opp", "seed", "a_first", "a_score", "b_score"} for r in rep["per_game"])


def test_resequence_contract_records_not_asserts(tmp_path, capsys):
    """Same multiset, different card order: record whether games are identical (no assert either way)."""
    from ygo_zero.decks import load_ydk

    mod = _mod()
    d = load_ydk(FIX / "BlueEyes.ydk")
    reseq = tmp_path / "BlueEyesReseq.ydk"
    _write_ydk(reseq, list(reversed(d.main)), list(reversed(d.extra)))
    out = tmp_path / "ab"
    rc = _run(mod, FIX / "BlueEyes.ydk", reseq, out)
    assert rc == 0, capsys.readouterr().out
    rep = json.loads((out / "report.json").read_text())
    assert rep["identical"] and not rep["same_order"]
    scores_equal = all(r["a_score"] == r["b_score"] for r in rep["per_game"])
    (tmp_path / "finding.txt").write_text(
        f"resequenced BlueEyes (reversed main+extra) vs original, greedy vs greedy, "
        f"{rep['n_pairs']} pairs: paired scores all equal = {scores_equal}\n")
    if not scores_equal:  # games differ -> the harness must say pairing is by seed, not by draws
        assert "card order changes deals: pairing is by seed, not by identical draws" in \
            (out / "summary.txt").read_text()
    print(f"resequence finding: paired scores all equal = {scores_equal}")


def test_duplicate_copies_play_and_pair(tmp_path, capsys):
    """3 copies of a card vs 2 copies + 1 other card: both legal, every game pairs."""
    from ygo_zero.decks import load_ydk

    mod = _mod()
    d = load_ydk(FIX / "BlueEyes.ydk")
    counts = Counter(d.main)
    double, single = next(c for c, n in counts.items() if n == 2), next(c for c, n in counts.items() if n == 1)
    main3, main2 = list(d.main), list(d.main)
    main3.remove(single)
    main3.append(double)  # double now at 3 copies, single removed; total count unchanged
    counts3 = Counter(main3)
    assert max(counts3.values()) <= 3, "test setup keeps the 3-copy legal limit"
    a, b = tmp_path / "three.ydk", tmp_path / "two.ydk"
    _write_ydk(a, main3, list(d.extra))
    _write_ydk(b, main2, list(d.extra))
    out = tmp_path / "ab"
    rc = _run(mod, a, b, out, ["--n-opp-decks", "1"])
    assert rc == 0, capsys.readouterr().out
    rep = json.loads((out / "report.json").read_text())
    assert rep["n_pairs"] == 1 * 1 * 2 == len(rep["per_game"])
