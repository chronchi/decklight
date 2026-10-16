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


def _tiny_value_and_aux(tmp_path: Path):
    """Random-init tiny value ckpt + matching aux.pt (CPU-only, no trained weights)."""
    import hashlib

    import torch

    from ygo_zero.eval.aux import AUX_SPEC, AuxHeads
    from ygo_zero.model.policy import PolicyNet

    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, ed = 50, 8
    emb_path = tmp_path / "tiny-emb.npz"
    np.savez(emb_path, codes=np.arange(1, n + 1),
             card=rng.normal(size=(n, ed)).astype(np.float32),
             effects=rng.normal(size=(n, 2, ed)).astype(np.float32),
             n_effects=np.ones(n, dtype=np.int64))
    emb = dict(np.load(emb_path))
    net = PolicyNet(emb, d=8, layers=1, heads=2, value=True)
    vdir = tmp_path / "vinit"
    vdir.mkdir(exist_ok=True)
    vckpt = vdir / "best.pt"
    torch.save({k: v for k, v in net.state_dict().items()}, vckpt)
    (vdir / "metrics.jsonl").write_text(
        json.dumps({"args": {"d": 8, "layers": 1, "heads": 2, "value": True}}) + "\n"
    )
    sha = hashlib.sha256(vckpt.read_bytes()).hexdigest()
    aux = AuxHeads(8)
    aux_path = tmp_path / "aux.pt"
    torch.save(
        {"model": {k: v.half() if v.is_floating_point() else v for k, v in aux.state_dict().items()},
         "aux_spec": AUX_SPEC, "value_sha256": sha},
        aux_path,
    )
    return emb_path, vckpt, aux_path


def _run_score_real(monkeypatch, games_dir: Path, out: Path, emb: Path, vckpt: Path, extra=()):
    import render_showcase as R

    fixtures = ROOT / "tests" / "fixtures" / "decks"
    monkeypatch.setattr(R, "deck_path", lambda n: fixtures / f"{n}.ydk")
    old = sys.argv
    sys.argv = [
        "score_decisions", "--games", str(games_dir), "--value", str(vckpt),
        "--emb", str(emb), "--out", str(out), "--batch", "16", "--device", "cpu",
        *extra,
    ]
    try:
        assert score_decisions.main() == 0
    finally:
        sys.argv = old
    return [json.loads(x) for x in out.read_text().splitlines() if x.strip()]


def test_aux_adds_aligned_arrays_and_leaves_v_identical(engine, decks, tmp_path, monkeypatch):
    emb, vckpt, auxpt = _tiny_value_and_aux(tmp_path)
    games_dir = _write_games_dir(tmp_path, engine, decks, n_games=2)
    plain = _run_score_real(monkeypatch, games_dir, tmp_path / "plain.jsonl", emb, vckpt)
    assert all("aux" not in r for r in plain)
    scored = _run_score_real(
        monkeypatch, games_dir, tmp_path / "aux.jsonl", emb, vckpt, ("--aux", str(auxpt)))
    assert len(scored) == len(plain)
    for p, s in zip(plain, scored, strict=True):
        assert s["k"] == p["k"] and s["v"] == p["v"]  # trunk value head equals today's v
        ax = s["aux"]
        n = len(s["k"])
        assert len(ax["d_lp"]) == len(ax["d_cards"]) == len(ax["ends"]) == len(ax["turns_left"]) == n
        for pair in ax["d_lp"]:
            assert len(pair) == 2 and all(float(x) == int(x) for x in pair)
        for row in ax["d_cards"]:
            assert len(row) == 14 and all(x == pytest.approx(round(x, 2)) for x in row)
        for e in ax["ends"]:
            assert 0.0 <= e <= 1.0 and e == pytest.approx(round(e, 3))
        for t in ax["turns_left"]:
            assert t == pytest.approx(round(t, 2))
        assert all(np.isfinite(s["v"]))


def test_aux_sha_mismatch_exits_2(engine, decks, tmp_path, monkeypatch):
    import torch

    from ygo_zero.eval.aux import AUX_SPEC, AuxHeads

    emb, vckpt, _ = _tiny_value_and_aux(tmp_path)
    bad = tmp_path / "aux-bad.pt"
    torch.save({"model": {k: v for k, v in AuxHeads(8).state_dict().items()},
                "aux_spec": AUX_SPEC, "value_sha256": "0" * 64}, bad)
    games_dir = _write_games_dir(tmp_path, engine, decks, n_games=1)
    import render_showcase as R

    fixtures = ROOT / "tests" / "fixtures" / "decks"
    monkeypatch.setattr(R, "deck_path", lambda n: fixtures / f"{n}.ydk")
    old = sys.argv
    sys.argv = ["score_decisions", "--games", str(games_dir), "--value", str(vckpt),
                "--emb", str(emb), "--out", str(tmp_path / "o.jsonl"),
                "--aux", str(bad), "--device", "cpu"]
    try:
        with pytest.raises(SystemExit) as e:
            score_decisions.main()
        assert e.value.code == 2
    finally:
        sys.argv = old
