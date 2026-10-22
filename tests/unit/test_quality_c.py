"""Task C: judge weights/turn-interrupt matching (fake record, no engine/LLM) + narrative builder."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_judge = _load("judge_games")

from decklight.demos import dataset as _ds


def _obs():
    g = np.zeros(24, np.int32)
    g[2] = 1
    return {"x": np.zeros(3, np.int32), "global": g}


class _FakeEnv:
    n = 3

    def __init__(self, *a):
        self.i = 0
        self.player = 0

    def reset(self):
        return _obs()

    def step(self, ch):
        self.i += 1
        return _obs() if self.i < _FakeEnv.n else None

    def history_len(self):
        return 0

    def history_log(self):
        return np.zeros((0, 14), np.int32)


def _setup_fake(monkeypatch, tmp_path, rec_extra=None, judge=None):
    rec = {
        "game_id": "game1",
        "line_id": "a-vs-b/game1",
        "decks": ["a", "b"],
        "config": {"seed": 7},
        "choices": [0, 0, 0],
        "labels": ["stated", "interrupt", "inferred"],
        "actors": [0, 1, 0],
        "turns": [1, 1, 2],
    }
    if rec_extra:
        rec.update(rec_extra)
    src = tmp_path / "a-vs-b.jsonl"
    src.write_text(json.dumps(rec) + "\n")
    monkeypatch.setattr(_ds, "CppEnv", _FakeEnv)
    monkeypatch.setattr(_ds, "record_config", lambda rec, seed=None: object())
    monkeypatch.setattr(_ds, "JUDGED_DIR", tmp_path / "judged")
    _ds._judge_cache.clear()
    _FakeEnv.n = len(rec["choices"])
    if judge is not None:
        (tmp_path / "judged").mkdir(parents=True, exist_ok=True)
        (tmp_path / "judged" / src.name).write_text(json.dumps(judge) + "\n")
    return src


def test_bad_turn_dropped_mediocre_kept_interrupt_matched(tmp_path, monkeypatch):
    judge = {
        "game_id": "game1",
        "line_id": "a-vs-b/game1",
        "turns": {"1": {"score": 2, "why": "bad"}, "2": {"score": 5, "why": "great"}},
        "interrupts": [{"turn": 1, "player": 1, "card": "Ash", "score": 4, "why": "ok"}],
    }
    src = _setup_fake(monkeypatch, tmp_path, judge=judge)
    got = list(_ds.iter_samples(None, src))
    assert [(m["k"], w, m["judge"]) for _, _, w, m in got] == [(1, 1.0, 4), (2, 0.5, 5)]


def test_interrupts_matched_in_order_per_turn_player(tmp_path, monkeypatch):
    judge = {
        "game_id": "game1",
        "line_id": "a-vs-b/game1",
        "turns": {},
        "interrupts": [
            {"turn": 3, "player": 1, "card": "A", "score": 5, "why": ""},
            {"turn": 3, "player": 1, "card": "B", "score": 2, "why": ""},
        ],
    }
    src = _setup_fake(
        monkeypatch,
        tmp_path,
        rec_extra={"labels": ["interrupt", "interrupt", "stated"], "actors": [1, 1, 0], "turns": [3, 3, 4]},
        judge=judge,
    )
    got = list(_ds.iter_samples(None, src))
    assert [(m["k"], w, m["judge"]) for _, _, w, m in got] == [(0, 1.0, 5), (2, 1.0, None)]


def test_no_judgement_unchanged_weight(tmp_path, monkeypatch):
    src = _setup_fake(monkeypatch, tmp_path)
    got = list(_ds.iter_samples(None, src))
    assert [(m["k"], w, m["judge"]) for _, _, w, m in got] == [(0, 1.0, None), (1, 1.0, None), (2, 0.5, None)]


def test_filter_interrupts_drops_invented():
    rec = {"interrupts": [{"turn": 2, "player": 1, "card": "Ash", "why": "x"}]}
    rated = [
        {"turn": 2, "player": 1, "card": "Ash", "score": 4, "why": "ok"},
        {"turn": 2, "player": 1, "card": "Ash", "score": 2, "why": "dup"},
        {"turn": 5, "player": 0, "card": "X", "score": 5, "why": "invented"},
    ]
    assert _judge.filter_interrupts(rated, rec) == [rated[0]]
    assert _judge.filter_interrupts(rated, {}) == []


def test_turn_decks_prefers_plan_owner_over_first_actor():
    rec = {
        "decks": ["a", "b"],
        "turns": [2, 2, 2],
        "actors": [0, 1, 1],
        "plans": [{"turn": 2, "player": 1, "deck": "b", "why": "turn", "attempts": []}],
    }
    assert _judge.turn_decks(rec) == {"2": "b"}
    rec2 = {"decks": ["a", "b"], "turns": [3, 3, 3], "actors": [0, 1, 1], "plans": []}
    assert _judge.turn_decks(rec2) == {"3": "b"}


def test_parse_judgement_valid_fenced_and_garbage():
    good = _judge.parse_judgement(
        '```json\n{"turns": {"1": {"score": 4, "why": "ok"}}, '
        '"interrupts": [{"turn": 1, "player": 1, "card": "Ash", "score": 2, "why": "x"}]}```'
    )
    assert good["turns"] == {"1": {"score": 4, "why": "ok"}}
    assert good["interrupts"][0]["score"] == 2
    bad = _judge.parse_judgement("not json at all {{{")
    assert bad == {"turns": {}, "interrupts": []}
    odd = _judge.parse_judgement('{"turns": {"1": {"score": 9}, "x": {"score": 3}}, "interrupts": "nope"}')
    assert odd["turns"] == {} and odd["interrupts"] == []


@pytest.mark.skipif(
    not (ROOT / "data/demos/games/blue-eyes-vs-altergeist.jsonl").exists(),
    reason="game record absent (data/ is gitignored)",
)
def test_narrative_on_tiny_real_record():
    from decklight.demos.names import CardNames
    from decklight.engine.duel import Engine

    rec = json.loads((ROOT / "data/demos/games/blue-eyes-vs-altergeist.jsonl").read_text().splitlines()[0])
    for k in ("choices", "labels", "actors", "turns"):
        rec[k] = rec[k][:12]
    text = _judge.build_narrative(rec, Engine(), CardNames())
    assert "Turn 1" in text and "blue-eyes" in text and "Hand:" in text
    assert len(text) < 12000
