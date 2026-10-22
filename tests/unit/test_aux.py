"""Aux evaluator heads N3 (src/decklight/eval/aux.py, scripts/build_aux_shards.py,
scripts/train_aux.py): target extraction on hand-made sequences, features() leaving
forward_value unchanged (random-init tiny model), AuxHeads shapes/losses, and a
2-epoch train_aux smoke run on stub games with a tiny random model.

No trained-model inference: only random-init tiny models and fake/stub games
(allowed: tiny/code-only tests, no model inference).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"

sys.path.insert(0, str(ROOT / "src"))

from decklight.eval.aux import AUX_SPEC, AuxHeads, aux_losses, extract_targets


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build_aux_shards = _load("build_aux_shards")
train_aux = _load("train_aux")

ME = [30, 5, 0, 0, 0, 0, 10]
OP = [30, 5, 0, 0, 0, 0, 10]
ME2 = [28, 4, 1, 0, 1, 0, 10]


def _g(lp_me, lp_opp, turn, my_turn, me_c, op_c):
    return [lp_me, lp_opp, turn, 0, 0, my_turn] + me_c + op_c + [0, 0, 0, 1]


def _game():
    """5 decisions: seat 0 turn 1, seat 1 answers, seat 1 turn 2, seat 0 answers, seat 0 turn 3."""
    G = np.array([
        _g(8000, 8000, 1, 1, ME, OP),
        _g(8000, 8000, 1, 0, ME, OP),
        _g(8000, 7500, 2, 1, ME2, OP),
        _g(7500, 8000, 2, 0, ME, ME2),
        _g(7500, 7000, 3, 1, ME2, ME2),
    ], dtype=np.int32)
    return G, np.array([0, 1, 1, 0, 0])


def test_extract_targets_own_turn_and_masks():
    G, a = _game()
    t = extract_targets(G, a, 0, 3)
    assert t["mask"].tolist() == [True, True, False, True, False]
    assert t["ends"].tolist() == [0.0, 0.0, 1.0, 0.0, 1.0]
    # k0 (seat 0, turn 1) -> next own start k4: me -500, opp -1000
    assert t["d_lp"][0].tolist() == pytest.approx([-0.0625, -0.125])
    # k1 (seat 1, turn 1) -> next own start k2: me +0, opp -500
    assert t["d_lp"][1].tolist() == pytest.approx([0.0, -0.0625])
    # k3 (seat 0, turn 2) -> k4: me +0, opp -1000
    assert t["d_lp"][3].tolist() == pytest.approx([0.0, -0.125])
    # masked-out rows carry zeros
    assert (t["d_lp"][2] == 0).all() and (t["d_lp"][4] == 0).all()
    assert (t["d_cards"][2] == 0).all() and (t["d_cards"][4] == 0).all()
    # k0 card deltas: me ME->ME2, opp OP->ME2
    assert t["d_cards"][0].tolist() == pytest.approx([-2, -1, 1, 0, 1, 0, 0] * 2)
    # turns_left = log1p(end_turn - turn)
    assert t["turns_left"].tolist() == pytest.approx(
        [np.log1p(2), np.log1p(2), np.log1p(1), np.log1p(1), 0.0])
    assert t["d_lp"].shape == (5, 2) and t["d_cards"].shape == (5, 14)


def test_extract_targets_rejects_truncated():
    G, a = _game()
    with pytest.raises(AssertionError):
        extract_targets(G, a, 3, 3)


def test_aux_spec_covers_four_groups():
    assert set(AUX_SPEC) == {"d_lp", "d_cards", "ends", "turns_left"}
    assert AUX_SPEC["d_lp"]["dim"] == 2 and AUX_SPEC["d_cards"]["dim"] == 14
    assert AUX_SPEC["ends"]["loss"] == "bce" and AUX_SPEC["turns_left"]["baseline"] == "train_mean"


def _tiny_emb(path: Path):
    rng = np.random.default_rng(0)
    n, d = 50, 8
    np.savez(path, codes=np.arange(1, n + 1), card=rng.normal(size=(n, d)).astype(np.float32),
             effects=rng.normal(size=(n, 2, d)).astype(np.float32), n_effects=np.ones(n, dtype=np.int64))


def _real_obs_batch(engine, decks, n: int = 4):
    """A small batch of real engine obs (random play), stacked like shard rows."""
    import random

    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import DuelConfig
    from decklight.model.bc_data import OBS_KEYS

    names = sorted(decks)
    env = CppEnv(engine, DuelConfig(decks[names[0]], decks[names[1]], seed=7))
    obs = env.reset()
    rng = random.Random(7)
    rows = []
    while obs is not None and len(rows) < n:
        rows.append({k: obs[k].copy() for k in OBS_KEYS})
        obs = env.step(rng.randrange(int(obs["action_mask"].sum())))
    assert len(rows) == n
    return {k: np.stack([r[k] for r in rows]) for k in OBS_KEYS}


def test_features_leaves_forward_value_unchanged(engine, decks, tmp_path):
    from decklight.model.policy import PolicyNet

    torch.set_num_threads(1)
    torch.manual_seed(0)
    _tiny_emb(tmp_path / "emb.npz")
    net = PolicyNet(dict(np.load(tmp_path / "emb.npz")), d=8, layers=1, heads=2, value=True)
    net.eval()
    batch = _real_obs_batch(engine, decks)
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in batch.items()}
    with torch.no_grad():
        _, v1 = net.forward_value(t)
        f = net.features(t)
        _, v2 = net.forward_value(t)
    assert torch.equal(v1, v2)
    assert f.shape == (len(batch["global"]), 8) and torch.isfinite(f).all()


def test_aux_heads_shapes_and_losses():
    torch.manual_seed(0)
    aux = AuxHeads(32)
    x = torch.randn(5, 32)
    pred = aux(x)
    assert pred["d_lp"].shape == (5, 2) and pred["d_cards"].shape == (5, 14)
    assert pred["ends"].shape == (5,) and pred["turns_left"].shape == (5,)
    tgt = {"d_lp": torch.randn(5, 2), "d_cards": torch.randn(5, 14),
           "ends": torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
           "turns_left": torch.randn(5)}
    loss = aux_losses(pred, tgt, torch.tensor([True, True, False, True, False]))
    assert set(loss) == {"d_lp", "d_cards", "ends", "turns_left", "total"}
    assert all(torch.isfinite(v).all() for v in loss.values())


def _stub_games(tmp_path: Path) -> Path:
    gen_selfplay = _load("gen_selfplay")

    decks = FIXTURES / "decks"
    assert decks.is_dir()
    out = tmp_path / "sp"
    old = sys.argv
    sys.argv = ["gen_selfplay", "--policy", "stub", "--out", str(out), "--games", "8",
                "--league", "self:1.0", "--decks", str(decks), "--seed", "0", "--threads", "1"]
    try:
        gen_selfplay.main()
    finally:
        sys.argv = old
    files = sorted(out.glob("*.jsonl"))
    assert files
    recs = [json.loads(x) for f in files for x in f.read_text().splitlines() if x.strip()]
    assert any(r["winner"] in (0, 1) for r in recs)  # at least one finished game
    return out


def _tiny_value_ckpt(tmp_path: Path) -> Path:
    from decklight.model.policy import PolicyNet

    torch.set_num_threads(1)
    torch.manual_seed(0)
    net = PolicyNet(dict(np.load(tmp_path / "emb.npz")), d=32, layers=1, heads=2, value=True)
    ckpt = tmp_path / "vinit" / "best.pt"
    ckpt.parent.mkdir()
    torch.save({k: v for k, v in net.state_dict().items()}, ckpt)
    (ckpt.parent / "metrics.jsonl").write_text(json.dumps({"args": {"d": 32, "layers": 1, "heads": 2}}) + "\n")
    return ckpt


def test_train_aux_smoke(engine, tmp_path):
    from decklight.model.bc_data import OBS_KEYS

    _tiny_emb(tmp_path / "emb.npz")
    rec_dir = _stub_games(tmp_path)
    shards = tmp_path / "ashards"
    old = sys.argv
    sys.argv = ["build_aux_shards", "--records", str(rec_dir / "*.jsonl"), "--out", str(shards),
                "--stride", "1", "--jobs", "1"]
    try:
        build_aux_shards.main()
    finally:
        sys.argv = old
    files = sorted(shards.glob("*.npz"))
    assert files
    p = dict(np.load(files[0]))
    for k in OBS_KEYS:
        assert k in p
    for k in ("d_lp", "d_cards", "ends", "turns_left", "mask", "game", "actor", "game_ids", "game_row"):
        assert k in p, k
    assert len(p["game"]) > 0 and p["mask"].sum() > 0

    out = tmp_path / "aux"
    sys.argv = ["train_aux", "--shards", str(shards), "--emb", str(tmp_path / "emb.npz"),
                "--value", str(_tiny_value_ckpt(tmp_path)), "--out", str(out),
                "--epochs", "2", "--batch", "8", "--device", "cpu", "--threads", "1", "--seed", "0"]
    try:
        train_aux.main()
    finally:
        sys.argv = old
    assert (out / "aux.pt").is_file() and (out / "metrics.json").is_file()
    m = json.loads((out / "metrics.json").read_text())
    assert m["value_identical"] is True and m["beats_baseline"] is True
    for k in ("d_lp", "d_cards", "ends", "turns_left"):
        assert k in m["holdout"] and np.isfinite(m["holdout"][k]["mse"])
    ckpt = torch.load(out / "aux.pt", map_location="cpu", weights_only=False)
    assert set(ckpt["aux_spec"]) == {"d_lp", "d_cards", "ends", "turns_left"}
    assert len(ckpt["value_sha256"]) == 64
