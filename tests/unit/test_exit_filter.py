"""exit_filter: clean (z) + turn + per-game filtering of teacher records."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location("exit_filter", ROOT / "scripts" / "exit_filter.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["exit_filter"] = m
    spec.loader.exec_module(m)
    return m


def _rec():
    clean = [[1.0, 0.9, 1.1, 1.0], [0.0, 0.1, -0.1, 0.0]]  # gap 1, tiny noise
    noisy = [[1.0, -1.0, 1.0, -1.0], [0.0, 0.5, -0.5, 0.0]]
    search = [{"k": 0, "cands": [1, 0], "chosen": 1, "played": 0, "samples": clean},
              {"k": 1, "cands": [1, 0], "chosen": 1, "played": 1, "samples": noisy},
              {"k": 2, "cands": [0, 2], "chosen": 0, "played": 0, "samples": clean},
              {"k": 3, "cands": [0, 2], "chosen": 0, "played": 0, "samples": clean}]
    return {"choices": [1, 1, 0, 0, 0], "turns": [1, 1, 2, 3, 3], "labels": ["own"] * 4 + ["skip"],
            "targets": [1, 1, 0, 0, None], "search": search}


def test_filter_keeps_clean_early_labels(tmp_path, monkeypatch, capsys):
    m = _mod()
    assert m.clean_z([[1, 1], [0, 0]]) == float("inf")
    (tmp_path / "r.jsonl").write_text(json.dumps(_rec()) + "\n")
    monkeypatch.setattr("sys.argv", ["x", "--records", str(tmp_path / "r.jsonl"), "--out", str(tmp_path / "o"),
                                     "--per-game", "5", "--max-turn", "2"])
    m.main()
    r = json.loads((tmp_path / "o/r.jsonl").read_text())
    assert [e["k"] for e in r["search"]] == [0, 2]  # 1 noisy, 3 past max turn
    assert r["labels"] == ["own", "skip", "own", "skip", "skip"] and r["targets"] == [1, None, 0, None, None]
    out = capsys.readouterr().out
    assert "unclean 1" in out and "turn 1" in out and "kept 2" in out


def test_per_game_cap(tmp_path, monkeypatch):
    m = _mod()
    (tmp_path / "r.jsonl").write_text(json.dumps(_rec()) + "\n")
    monkeypatch.setattr("sys.argv", ["x", "--records", str(tmp_path / "r.jsonl"), "--out", str(tmp_path / "o"),
                                     "--per-game", "1", "--max-turn", "0"])
    m.main()
    r = json.loads((tmp_path / "o/r.jsonl").read_text())
    assert len(r["search"]) == 1 and sum(lab != "skip" for lab in r["labels"]) == 1
