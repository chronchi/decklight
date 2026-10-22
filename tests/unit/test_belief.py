"""Belief head (PLAN §0a "Hidden-information bound" (b)): shards, baseline, model, trainer.

Only tiny d=16 models are built here (RAM rule).
"""

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_value import _load, _tiny_emb, _write_games

build_belief_shards = _load("build_belief_shards")
train_belief = _load("train_belief")


def test_hypergeom_matches_enumeration():
    for N, n_c, h in [(10, 2, 3), (7, 3, 7), (5, 1, 1), (6, 0, 2), (4, 3, 2)]:
        pool = [1] * n_c + [0] * (N - n_c)
        combos = list(itertools.combinations(range(N), h))
        want = sum(any(pool[i] for i in c) for c in combos) / len(combos)
        assert math.isclose(build_belief_shards.hypergeom_in(N, n_c, h), want, abs_tol=1e-9)


def _obs(rows, hand_codes_true):
    """Synthetic obs: rows = (code, loc, ctrl, owner, card_known, known_code); priv_code from hand_codes_true."""
    n = 160
    cards = np.zeros((n, 21), np.int32)
    mask = np.zeros(n, bool)
    known = np.zeros(n, np.int8)
    kcode = np.zeros(n, np.int32)
    priv = np.zeros(n, np.int32)
    it = iter(hand_codes_true)
    for i, (code, loc, ctrl, owner, k, kc) in enumerate(rows):
        cards[i, 0], cards[i, 1], cards[i, 3], cards[i, 4] = code, loc, ctrl, owner
        mask[i], known[i], kcode[i] = True, k, kc
        if ctrl == 1 and code == 0:
            priv[i] = next(it)
    return {"cards": cards, "card_mask": mask, "card_known": known, "known_code": kcode, "priv_code": priv}


def test_belief_target_excludes_seen_and_known():
    # opponent list: 3x A(10), 2x B(20), 1x C(30). Visible: one A in the opponent's grave. Hidden: 2 hand cards
    # (truth B and C), one of them remembered (card_known 1 -> C), and 1 face-down spell (truth A).
    rows = [(10, 5, 1, 1, 0, 0), (0, 2, 1, 0, 0, 0), (0, 2, 1, 0, 1, 30), (0, 4, 1, 0, 0, 0), (55, 3, 0, 0, 0, 0)]
    obs = _obs(rows, [20, 30, 10])
    cand, cand_n, y, base, h, ul = build_belief_shards.belief_target(obs, [10, 10, 10, 20, 20, 30])
    assert h == 2 and ul == 0  # the remembered card is not hidden
    assert cand[:3].tolist() == [10, 20, 30] and cand_n[:3].tolist() == [3, 2, 1] and cand[3] == 0
    assert y[:3].tolist() == [1, 1, 0]
    # pool: A 2 (3 - grave), B 2, C 0 (remembered) -> N 4, h 2
    assert base[2] == 0.0
    assert math.isclose(base[0], 5 / 6, rel_tol=1e-6) and math.isclose(base[1], 5 / 6, rel_tol=1e-6)


def _build(src: Path, tmp_path: Path) -> Path:
    out = tmp_path / "bshards"
    old = sys.argv
    sys.argv = ["build_belief_shards", "--records", str(src), "--out", str(out), "--stride", "2", "--jobs", "1",
                "--hist", "16"]
    try:
        build_belief_shards.main()
    finally:
        sys.argv = old
    return out


def test_belief_shards_labels_match_priv(engine, tmp_path):
    src = _write_games(engine, tmp_path)
    out = _build(src, tmp_path)
    p = dict(np.load(next(out.glob("*.npz"))))
    n = len(p["y"])
    assert n > 0 and p["hist_long"].shape[1:] == (16, 13)
    m = p["cand"] > 0
    assert (p["y"][~m] == 0).all() and (p["base_p"][~m] == 0).all()
    assert ((p["y"] == 1) & (p["base_p"] == 0)).sum() == 0  # a hidden card always has a nonzero baseline
    # every state has >= 1 hidden card and labels never exceed it
    assert (p["n_hidden"] >= 1).all() and (p["y"].sum(1) <= p["n_hidden"]).all()
    # calibration of the baseline on the scored entries: mean prediction ~ positive rate
    s = m & (p["base_p"] > 0) & (p["base_p"] < 1)
    assert abs(p["y"][s].mean() - p["base_p"][s].mean()) < 0.1


def _tiny_init(tmp_path: Path, emb: Path) -> Path:
    from decklight.model.policy import PolicyNet

    torch.manual_seed(0)
    net = PolicyNet(dict(np.load(emb)), d=16, layers=1, heads=2, dropout=0.0)
    d = tmp_path / "init"
    d.mkdir()
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()}, d / "best.pt")
    (d / "metrics.jsonl").write_text(json.dumps({"args": {"d": 16, "layers": 1, "heads": 2, "hist": 0,
                                                          "known": False, "holdout": ["tb1"]}}) + "\n")  # fmt: skip
    return d / "best.pt"


def test_init_trunk_widens_and_zeroes(tmp_path):
    from decklight.model.belief import BeliefNet, init_trunk
    from decklight.model.policy import PolicyNet

    emb_p = tmp_path / "emb.npz"
    _tiny_emb(emb_p)
    emb = dict(np.load(emb_p))
    torch.manual_seed(0)
    src = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0)
    trunk = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, hist=8, known=True)
    fresh = init_trunk(trunk, src.state_dict())
    assert set(fresh) == {"known_emb.weight", "hist_pos.weight"}
    assert torch.equal(trunk.hist_age.weight[:8], src.hist_age.weight)
    assert torch.equal(trunk.hist_age.weight[8:], src.hist_age.weight[-1:].expand(24, -1))
    assert not trunk.known_emb.weight.any() and not trunk.hist_pos.weight.any()
    net = BeliefNet(trunk)
    assert not net.mlp[-1].weight.any()


def _train(shards: Path, init: Path, emb: Path, out: Path, *extra: str) -> None:
    old = sys.argv
    sys.argv = ["train_belief", "--shards", str(shards), "--init", str(init), "--emb", str(emb), "--out", str(out),
                "--epochs", "1", "--batch", "16", "--eval-batch", "32", "--evals-per-epoch", "1", "--val-mod", "3", *extra]
    try:
        train_belief.main()
    finally:
        sys.argv = old


@pytest.mark.parametrize("extra", [(), ("--hist", "16"), ("--nohist", "--freeze-trunk")])
def test_train_belief_runs_and_reports(engine, tmp_path, extra):
    torch.set_num_threads(1)
    src = _write_games(engine, tmp_path)
    shards = _build(src, tmp_path)
    emb = tmp_path / "emb.npz"
    _tiny_emb(emb)
    init = _tiny_init(tmp_path, emb)
    out = tmp_path / "run" / "v"
    _train(shards, init, emb, out, *extra)
    recs = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    assert recs[0]["splits"]["train"] > 0
    assert recs[-1]["final"] and np.isfinite(recs[-1]["val_ratio"])
    assert (out / "final.pt").is_file()
