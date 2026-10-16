"""Unit tests for scripts/score_decisions.py + the --values annotation path.

No torch, no trained model: scoring runs with a fake value object returning a
deterministic function of obs, on a short random engine game built with the
conftest fixtures (allowed: tiny/code-only tests, no model inference).
"""

import importlib.util
import json
import random
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


score_decisions = _load("score_decisions")


def test_parse_shard():
    assert score_decisions.parse_shard("0/1") == (0, 1)
    assert score_decisions.parse_shard("2/4") == (2, 4)
    for bad in ("4/4", "0/0", "x", "1/2/3", "-1/2"):
        with pytest.raises(SystemExit) as e:
            score_decisions.parse_shard(bad)
        assert e.value.code == 2


class FakeValue:
    """Deterministic actor-view value: LP lead / 16000 (finite, varies, no model)."""

    def __init__(self, *a, **k):
        pass

    def values(self, buf: dict) -> np.ndarray:
        g = buf["global"]
        return np.clip((g[:, 0] - g[:, 1]) / 16000.0, -1, 1).astype(np.float32)


def _short_game(engine, decks, seed: int, max_steps: int = 60):
    names = sorted(decks)
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.engine.duel import DuelConfig

    env = CppEnv(engine, DuelConfig(decks[names[0]], decks[names[1]], seed=seed))
    obs = env.reset()
    rng = random.Random(seed)
    choices = []
    for _ in range(max_steps):
        if obs is None:
            break
        n = int(obs["action_mask"].sum())
        ch = rng.randrange(n)
        choices.append(ch)
        obs = env.step(ch)
    return names, choices


def _write_games_dir(tmp_path: Path, engine, decks, n_games: int = 3) -> Path:
    d = tmp_path / "games-in"
    d.mkdir()
    lines = []
    for i in range(n_games):
        names, choices = _short_game(engine, decks, seed=1000 + i)
        lines.append(
            json.dumps(
                {
                    "game_idx": i,
                    "decks": names[:2],
                    "seed": 1000 + i,
                    "s_seat": 0,
                    "winner": None,
                    "truncated": True,
                    "turns": 3,
                    "choices": choices,
                    "search": [],
                }
            )
        )
    (d / "games-0.jsonl").write_text("\n".join(lines) + "\n")
    return d


def _run_score(monkeypatch, games_dir: Path, out: Path, extra=()):
    import ygo_zero.eval.value as V

    monkeypatch.setattr(V, "ValueNet", FakeValue)
    import render_showcase as R

    fixtures = ROOT / "tests" / "fixtures" / "decks"
    # score_decisions resolves decks through render_showcase.deck_path (pool|gauntlet|newcards);
    # point it at the fixture decks instead.
    monkeypatch.setattr(R, "deck_path", lambda n: fixtures / f"{n}.ydk")
    old = sys.argv
    sys.argv = [
        "score_decisions",
        "--games",
        str(games_dir),
        "--value",
        "dummy.pt",
        "--emb",
        "dummy.npz",
        "--out",
        str(out),
        "--batch",
        "16",
        *extra,
    ]
    try:
        assert score_decisions.main() == 0
    finally:
        sys.argv = old
    return [json.loads(x) for x in out.read_text().splitlines() if x.strip()]


def test_score_every_decision(engine, decks, tmp_path, monkeypatch):
    games_dir = _write_games_dir(tmp_path, engine, decks)
    recs = [json.loads(x) for x in (games_dir / "games-0.jsonl").read_text().splitlines()]
    rows = _run_score(monkeypatch, games_dir, tmp_path / "values.jsonl")
    assert len(rows) == len(recs)
    for rec, row in zip(recs, rows, strict=True):
        assert row["game_idx"] == rec["game_idx"]
        assert row["seed"] == rec["seed"]
        assert row["decks"] == rec["decks"]
        assert row["s_seat"] == rec["s_seat"]
        assert row["k"] == list(range(len(rec["choices"])))
        assert len(row["actor"]) == len(rec["choices"])
        assert set(row["actor"]) <= {0, 1}
        assert len(row["turn"]) == len(rec["choices"])
        assert len(row["v"]) == len(rec["choices"])
        assert all(np.isfinite(row["v"]))
        # 4-decimal rounding
        for x in row["v"]:
            assert x == pytest.approx(round(x, 4))


def test_shard_split_covers_all(engine, decks, tmp_path, monkeypatch):
    games_dir = _write_games_dir(tmp_path, engine, decks, n_games=4)
    a = _run_score(monkeypatch, games_dir, tmp_path / "v0.jsonl", ("--shard", "0/2"))
    b = _run_score(monkeypatch, games_dir, tmp_path / "v1.jsonl", ("--shard", "1/2"))
    assert sorted(r["game_idx"] for r in a) == [0, 2]
    assert sorted(r["game_idx"] for r in b) == [1, 3]


def test_zero_games_exits_2(tmp_path, monkeypatch):
    import ygo_zero.eval.value as V

    monkeypatch.setattr(V, "ValueNet", FakeValue)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "games-0.jsonl").write_text("")
    old = sys.argv
    sys.argv = [
        "score_decisions",
        "--games",
        str(empty),
        "--value",
        "dummy.pt",
        "--out",
        str(tmp_path / "v.jsonl"),
    ]
    try:
        with pytest.raises(SystemExit) as e:
            score_decisions.main()
        assert e.value.code == 2
    finally:
        sys.argv = old
