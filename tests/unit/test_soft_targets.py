"""Gumbel-style soft targets from search labels (train_bc --soft-targets)."""

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


train_bc = _load("train_bc")


def _pi(ref, mask, si, sq, scale=2.0):
    return train_bc.gumbel_soft_target(ref, mask, si, sq, scale)


def test_equal_scores_give_prior():
    ref = torch.tensor([[0.5, -0.2, 1.0, float("-inf")]])
    mask = torch.tensor([[True, True, True, False]])
    si = torch.tensor([[0, 1, -1, -1, -1, -1, -1, -1]])
    sq = torch.tensor([[2.0, 2.0, 0, 0, 0, 0, 0, 0]])
    pi, has = _pi(ref, mask, si, sq)
    prior = torch.softmax(ref.masked_fill(~mask, float("-inf")), -1)
    torch.testing.assert_close(pi, prior, atol=1e-6, rtol=0)
    assert has.tolist() == [True]


def test_better_candidate_gains_exactly_exp_scale():
    import math

    ref = torch.tensor([[0.3, -0.7]])  # non-uniform prior: the shift is relative to it
    mask = torch.ones(1, 2, dtype=torch.bool)
    si = torch.tensor([[0, 1, -1, -1, -1, -1, -1, -1]])
    sq = torch.tensor([[10.0, 0.0, 0, 0, 0, 0, 0, 0]])
    pi, _ = _pi(ref, mask, si, sq, scale=2.0)
    prior = torch.softmax(ref, -1)
    got = (pi[0, 0] / pi[0, 1]) / (prior[0, 0] / prior[0, 1])
    assert got.item() == pytest.approx(math.exp(2.0), rel=1e-5)


def test_unsearched_options_get_v_mix():
    # independent numpy reference: completed Q = scores on cands, prior-weighted mean elsewhere
    rng = np.random.default_rng(0)
    ref = torch.tensor(rng.normal(size=(3, 5)).astype(np.float32))
    mask = torch.tensor([[True] * 4 + [False], [True] * 5, [True, True] + [False] * 3])
    si = torch.tensor([[0, 2, -1, -1, -1, -1, -1, -1],
                       [1, 3, 4, -1, -1, -1, -1, -1],
                       [0, -1, -1, -1, -1, -1, -1, -1]])
    sq = torch.tensor([[1.5, -0.5, 0, 0, 0, 0, 0, 0],
                       [0.3, 0.9, -1.2, 0, 0, 0, 0, 0],
                       [4.0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float16)
    pi, has = _pi(ref, mask, si, sq, scale=2.0)
    assert has.tolist() == [True, True, True]
    assert torch.isfinite(pi).all()
    assert (pi[~mask] == 0).all()  # illegal options get exactly 0
    ref_n = ref.numpy()
    for b in range(3):
        legal = mask[b].numpy()
        lp = ref_n[b].copy()
        lp[~legal] = -np.inf
        lp -= np.logaddexp.reduce(lp[legal])
        prior = np.exp(lp[legal])
        cands = [int(a) for a in si[b].numpy() if a >= 0]
        q = np.array([float(x) for x in sq[b].numpy()[: len(cands)]])
        vmix = float((prior[[c for c in cands]] * q).sum() / prior[cands].sum())
        full = np.full(5, vmix)
        for a, v in zip(cands, q):
            full[a] = v
        qh = np.zeros(5)
        qh[legal] = (full[legal] - full[legal].min()) / (full[legal].max() - full[legal].min() + 1e-6)
        imp = lp + 2.0 * qh
        imp[~legal] = -np.inf
        want = np.exp(imp - np.logaddexp.reduce(imp[legal]))
        want[~legal] = 0.0
        np.testing.assert_allclose(pi[b].numpy(), want, rtol=1e-4, atol=1e-6)


def test_no_search_info_gives_prior_and_no_soft():
    ref = torch.randn(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    si = torch.full((2, 8), -1)
    sq = torch.zeros(2, 8)
    pi, has = _pi(ref, mask, si, sq)
    torch.testing.assert_close(pi, torch.softmax(ref, -1), atol=1e-6, rtol=0)
    assert has.tolist() == [False, False]
    # None arrays (shards without soft fields) behave the same
    pi2, has2 = train_bc.gumbel_soft_target(ref, mask, None, None, 2.0)
    torch.testing.assert_close(pi2, torch.softmax(ref, -1), atol=1e-6, rtol=0)
    assert has2.tolist() == [False, False]


def test_illegal_candidates_are_ignored():
    ref = torch.zeros(1, 4)
    mask = torch.tensor([[True, True, False, False]])
    si = torch.tensor([[2, 3, -1, -1, -1, -1, -1, -1]])  # both illegal here
    sq = torch.tensor([[9.0, 9.0, 0, 0, 0, 0, 0, 0]])
    pi, has = _pi(ref, mask, si, sq)
    assert has.tolist() == [False]  # no legal search info -> hard-target row
    torch.testing.assert_close(pi, torch.tensor([[0.5, 0.5, 0.0, 0.0]]), atol=1e-6, rtol=0)


def test_search_soft_targets_selects_selection_score():
    from decklight.demos.dataset import search_soft_targets

    rec = {
        "source": "window search k=4 select=board@h1 tie_eps=0.0 over bc; relabelled select=board@h1",
        "search": [
            {"k": 3, "cands": [1, 0], "scores": [9.9, 9.9],
             "scores_all": {"board@h1": [0.2, 0.8], "value@h1": [0.7, 0.1]}},
            {"k": 4, "cands": [2], "scores": [0.5]},  # old format: scores are the selection scores
            {"k": 5, "cands": [0, 1], "scores": [0.1]},  # length mismatch -> skipped
            {"k": 6, "cands": [], "scores": []},  # empty -> skipped
        ],
    }
    got = search_soft_targets(rec)
    assert got[3] == ([1, 0], [0.2, 0.8])  # select=board@h1, not scores nor value@h1
    assert got[4] == ([2], [0.5])
    assert 5 not in got and 6 not in got
    # no select= in source: default value@h1, falling back to scores without scores_all
    rec2 = {"source": "window search k=4 h=1 over bc",
            "search": [{"k": 0, "cands": [1, 0], "scores": [-1.5, 0.0]}]}
    assert search_soft_targets(rec2) == {0: ([1, 0], [-1.5, 0.0])}


def _search_record(engine, tmp_path: Path, n_want: int = 6) -> Path:
    """One random game relabelled with `own` targets + search entries on decisions with >=3 options."""
    from decklight.decks import load_ydk
    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import DuelConfig

    decks = sorted((FIXTURES / "decks").glob("*.ydk"))
    d0, d1 = load_ydk(decks[0]), load_ydk(decks[1])
    cfg = DuelConfig(d0, d1, seed=50)
    env = CppEnv(engine, cfg)
    obs, ch, act, tu = env.reset(), [], [], []
    legals = []
    while obs is not None and len(ch) < 400:
        leg = np.flatnonzero(obs["action_mask"]).tolist()
        legals.append(leg)
        a = int(leg[0])
        ch.append(a)
        act.append(int(env.player))
        tu.append(int(obs["global"][2]))
        obs = env.step(a)
    conf = {"deck0_main": list(d0.main), "deck0_extra": list(d0.extra), "deck1_main": list(d1.main),
            "deck1_extra": list(d1.extra), "seed": cfg.seed, "hand": cfg.hand, "draw": cfg.draw,
            "lp": cfg.lp, "setup": [], "shuffle": cfg.shuffle}  # fmt: skip
    n = len(ch)
    labels = ["skip"] * n
    targets: list = [None] * n
    tsets: list = [None] * n
    tconf: list = [None] * n
    search = []
    got = 0
    for k in range(n):
        if len(legals[k]) >= 3 and got < n_want:
            labels[k] = "own"
            targets[k] = ch[k]
            tsets[k] = [ch[k]]
            tconf[k] = 1.0
            others = [a for a in legals[k] if a != ch[k]][:2]
            cands = [ch[k]] + others
            search.append({"k": k, "kind": "own", "turn": tu[k], "cands": cands,
                           "scores": [0.5, -0.2, 0.1][: len(cands)],
                           "scores_all": {"value@h1": [0.5, -0.2, 0.1][: len(cands)],
                                          "board@h1": [1.0, 2.0, 3.0][: len(cands)]},
                           "set": [ch[k]], "conf": 1.0, "chosen": ch[k], "played": ch[k], "fired": True})
            got += 1
    assert got == n_want, f"only {got} decisions with >=3 options"
    rec = {"game_id": "r0", "line_id": "a-vs-b/r0", "decks": ["a", "b"], "config": conf,
           "auto_single": True, "choices": ch, "actors": act, "turns": tu, "labels": labels,
           "targets": targets, "target_sets": tsets, "target_conf": tconf, "search": search,
           "source": "window search k=4 h=1 samples=4 select=value@h1 over bc; relabelled select=value@h1"}
    out = tmp_path / "sw-tiny" / "g.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(rec) + "\n")
    return out


def test_soft_fields_survive_record_to_shard(engine, tmp_path):
    from decklight.demos.dataset import iter_samples
    from decklight.model.bc_data import SOFT_K, _shard_one

    f = _search_record(engine, tmp_path)
    rec = json.loads(f.read_text().splitlines()[0])
    samples = [(o, a, w, m) for o, a, w, m in iter_samples(engine, f)]
    assert samples
    soft = [(o, a, w, m) for o, a, w, m in samples if "soft" in m]
    assert len(soft) == len(rec["search"])
    for obs, a, w, m in soft:
        s = next(s for s in rec["search"] if s["k"] == m["k"])
        cands, scores = m["soft"]
        assert cands == s["cands"] and scores == s["scores_all"]["value@h1"]  # cand/score alignment
        assert a == rec["targets"][m["k"]] == s["chosen"]
        mask = obs["action_mask"].astype(bool)
        assert all(mask[c] for c in cands)
    import decklight.model.bc_data as bcd

    old = bcd._ENGINE
    bcd._ENGINE = engine
    try:
        shard = tmp_path / "s.npz"
        _shard_one((str(f), str(shard), "sw-tiny"))
    finally:
        bcd._ENGINE = old
    z = dict(np.load(shard))
    assert z["soft_idx"].shape == (len(samples), SOFT_K)
    assert z["soft_q"].shape == (len(samples), SOFT_K)
    n_plain = sum(1 for _, _, _, m in samples if "soft" not in m)
    assert (z["soft_idx"] == -1).all(axis=1).sum() == n_plain
    # the highest-scored option wins the improved policy under a flat prior (highest q_hat)
    row = next(i for i in range(len(samples)) if (z["soft_idx"][i] >= 0).any())
    idx = z["soft_idx"][row]
    q = z["soft_q"][row]
    keep = idx >= 0
    best = int(idx[keep][np.argmax(q[keep].astype(np.float32))])
    mask = torch.ones(1, 64, dtype=torch.bool)
    pi, _ = _pi(torch.zeros(1, 64), mask, torch.from_numpy(idx[None, :]),
                torch.from_numpy(q[None, :]), scale=2.0)
    assert int(pi.argmax(-1)) == best
    assert best == rec["targets"][samples[row][3]["k"]]  # the argmax search label


def test_old_shards_without_soft_load_with_defaults(tmp_path):
    from decklight.model.bc_data import SOFT_K, load_shards

    d = tmp_path / "shards"
    d.mkdir()
    n = 5
    base = dict(action=np.arange(n), weight=np.ones(n), line=np.array([f"l{i}" for i in range(n)]),
                deck=np.array(["d0"] * n), kind=np.array(["sw-tiny_own"] * n), src=np.array(["t"] * n))
    np.savez_compressed(d / "old.npz", **base)  # no soft fields
    new = dict(base)
    new["soft_idx"] = np.tile(np.array([3, 1] + [-1] * (SOFT_K - 2), np.int16), (n, 1))
    new["soft_q"] = np.tile(np.array([0.5, -0.2] + [0.0] * (SOFT_K - 2), np.float16), (n, 1))
    np.savez_compressed(d / "new.npz", **new)
    data = load_shards(d)
    assert data["soft_idx"].shape == (2 * n, SOFT_K)
    # file order is not guaranteed: check row sets, not positions
    assert (data["soft_idx"] == -1).all(axis=1).sum() == n  # the old shard's rows
    assert (data["soft_idx"][:, 0] == 3).sum() == n  # the new shard's rows
    assert (data["soft_q"][data["soft_idx"][:, 0] == -1] == 0).all()


def _tiny_net(seed=0):
    from decklight.model.policy import PolicyNet

    rng = np.random.default_rng(42)
    codes = np.arange(1, 50)
    n = len(codes)
    emb = {"codes": codes, "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}
    torch.manual_seed(seed)
    return PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).train(), emb


def _tiny_obs(batch=8, seed=1, n_legal=4):
    from decklight.model.bc_data import OBS_KEYS

    torch.manual_seed(seed)
    shapes = {"cards": (batch, 160, 21), "card_mask": (batch, 160), "global": (batch, 24),
              "chain": (batch, 16, 5), "chain_mask": (batch, 16), "actions": (batch, 64, 10),
              "action_mask": (batch, 64), "history": (batch, 32, 13), "history_mask": (batch, 32)}
    obs = {}
    for k in OBS_KEYS:
        s = shapes[k]
        if k.endswith("_mask"):
            m = torch.zeros(s, dtype=torch.bool)
            m[:, :n_legal] = True
            obs[k] = m
        else:
            obs[k] = torch.randint(0, 4, s)
    obs["actions"] = torch.randint(0, 4, (batch, 64, 10))
    return obs


def test_gumbel_steps_are_finite_and_decrease():
    net, _ = _tiny_net()
    ref = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.SGD(net.parameters(), lr=3e-2)
    obs = _tiny_obs()
    mask = obs["action_mask"]
    si = torch.full((8, 8), -1)
    sq = torch.zeros(8, 8)
    si[:, 0] = 0
    si[:, 1] = 1  # searched candidates among the legal options
    sq[:, 0] = 1.0
    sq[:, 1] = -1.0
    first, last = None, None
    for _ in range(20):
        logits = net(obs)
        with torch.no_grad():
            pi_prime, has = _pi(ref(obs).float(), mask, si, sq)
        assert has.all() and torch.isfinite(pi_prime).all()
        logp = torch.log_softmax(logits.float().masked_fill(~mask, float("-inf")), dim=-1)
        loss = -(torch.where(mask, pi_prime, torch.zeros(())) *
                 torch.where(mask, logp, torch.zeros(()))).sum(-1).mean()
        assert torch.isfinite(loss)
        first = loss.item() if first is None else first
        last = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert last < first, f"gumbel loss did not decrease: {first:.4f} -> {last:.4f}"


def _synthetic_shards(path: Path, n: int = 64, seed: int = 0):
    """Tiny BC shards with soft info on half the samples (for the train_bc end-to-end test)."""
    from decklight.model.bc_data import SOFT_K

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_legal = 4
    data = {}
    shapes = {"cards": (n, 160, 21), "card_mask": (n, 160), "global": (n, 24),
              "chain": (n, 16, 5), "chain_mask": (n, 16), "actions": (n, 64, 10),
              "action_mask": (n, 64), "history": (n, 32, 13), "history_mask": (n, 32)}
    for k, s in shapes.items():
        data[k] = (rng.integers(0, 2, size=s).astype(bool) if k.endswith("_mask")
                   else rng.integers(0, 4, size=s).astype(np.int64))
        if k.endswith("_mask"):
            data[k][:, :n_legal] = True
    act = rng.integers(0, n_legal, size=n)
    eq = np.zeros((n, 64), dtype=bool)
    eq[np.arange(n), act] = True
    si = np.full((n, SOFT_K), -1, dtype=np.int16)
    sq = np.zeros((n, SOFT_K), dtype=np.float16)
    si[: n // 2, 0] = act[: n // 2]  # searched: the action + one runner-up, action scores best
    si[: n // 2, 1] = (act[: n // 2] + 1) % n_legal
    sq[: n // 2, 0] = 0.5
    sq[: n // 2, 1] = -0.5
    data.update(action=act.astype(np.int64), equiv=eq, weight=np.ones(n, np.float32),
                line=np.array([f"l{i % 16}" for i in range(n)]), deck=np.array(["d0"] * n),
                kind=np.array(["sw-tiny_own"] * n), src=np.array(["t"] * n),
                soft_idx=si, soft_q=sq)
    path.mkdir(parents=True)
    np.savez_compressed(path / "syn.npz", **data)


def _tiny_emb(path: Path):
    rng = np.random.default_rng(42)
    codes = np.arange(1, 50)
    n = len(codes)
    np.savez_compressed(path, codes=codes, card=rng.normal(size=(n, 8)).astype(np.float32),
                        effects=rng.normal(size=(n, 2, 8)).astype(np.float32),
                        n_effects=np.ones(n, np.int64))


def _run_bc(argv):
    old = sys.argv
    sys.argv = argv
    try:
        train_bc.main()
    finally:
        sys.argv = old


def test_gumbel_end_to_end_logs_soft(tmp_path):
    import json as _json

    shards = tmp_path / "shards"
    _synthetic_shards(shards)
    emb = tmp_path / "emb.npz"
    _tiny_emb(emb)
    out1 = tmp_path / "r1"
    _run_bc(["train_bc", "--shards", str(shards), "--emb", str(emb), "--out", str(out1),
             "--epochs", "1", "--batch", "8", "--d", "16", "--layers", "1", "--heads", "2",
             "--threads", "2", "--seed", "0", "--select-on", "last"])
    assert (out1 / "best.pt").is_file()
    out2 = tmp_path / "r2"
    _run_bc(["train_bc", "--shards", str(shards), "--emb", str(emb), "--out", str(out2),
             "--epochs", "1", "--batch", "8", "--threads", "2", "--seed", "0",
             "--init", str(out1 / "best.pt"), "--soft-targets", "gumbel", "--select-on", "last"])
    recs = [_json.loads(x) for x in (out2 / "metrics.jsonl").read_text().splitlines() if x.strip()]
    ep0 = next(r for r in recs if r.get("epoch") == 0)
    assert ep0["soft_n"] > 0
    assert np.isfinite(ep0["train_loss"])
    assert 0.0 < ep0["soft_ent"] <= np.log(64)
    assert 0.0 <= ep0["soft_argmax_agree"] <= 1.0


def test_gumbel_without_init_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_bc.py", "--shards", "x", "--emb", "y", "--out", "z",
                                      "--soft-targets", "gumbel"])
    with pytest.raises(SystemExit) as e:
        train_bc.main()
    assert e.value.code != 0


def test_soft_norm_none_keeps_near_ties_soft():
    """Review: min-max stretches a 0.01 score gap to the full scale c; norm='none' shifts by exp(c * gap) only."""
    ref = torch.zeros(1, 4)
    mask = torch.tensor([[True, True, True, False]])
    si = torch.tensor([[0, 1, -1]])
    sq = torch.tensor([[0.51, 0.50, 0.0]])
    pi, has = train_bc.gumbel_soft_target(ref, mask, si, sq, 5.0, "none")
    assert bool(has[0]) and pi[0, 3] == 0
    assert torch.isclose(pi[0, 0] / pi[0, 1], torch.tensor(math.exp(5.0 * 0.01)), rtol=1e-4)
    pm, _ = train_bc.gumbel_soft_target(ref, mask, si, sq, 5.0, "minmax")
    assert pm[0, 0] / pm[0, 1] > 100 * (pi[0, 0] / pi[0, 1])
