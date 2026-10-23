"""Collector (async rollout) over tiny PolicyNets on the fixture decks (CPU, fp32)."""

import copy
import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from decklight.decks import load_ydk
from decklight.engine.cpp_env import VecEnv
from decklight.model.policy import PolicyNet
from decklight.rl.ppo import SlotStore, gather, masked_log_softmax, to_torch
from decklight.rl.rollout import Collector

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"

_TRAIN_RL = None


def _train_rl_mod():
    global _TRAIN_RL
    if _TRAIN_RL is None:
        p = Path(__file__).resolve().parents[2] / "scripts" / "train_rl.py"
        spec = importlib.util.spec_from_file_location("train_rl_overlap_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _TRAIN_RL = mod
    return _TRAIN_RL


def _tiny(decks, value: bool):
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    return PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, value=value).eval()


def _collector(engine, decks, p_bc, seed=0):
    dev = torch.device("cpu")
    model, ref = _tiny(decks, True), _tiny(decks, False)
    vec = VecEnv(engine, decks, 8, 2, seed=seed)
    vec.set_max_decisions(60)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False))
    col = Collector(vec, model, ref, store, 0, False, dev, p_bc, -0.5, 2,
                    np.random.default_rng(seed), "fp32")
    return col, store, ref


def test_collect_finished_samples_and_cached_ref(engine):
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    col, store, ref = _collector(engine, decks, 0.3)
    finished, st = col.collect(500)
    total = sum(len(s) for s, _ in finished)
    assert total >= 500
    assert st["games"] > 0 and st["decisions"] >= total
    for _, rew in finished:
        assert rew in (1.0, -1.0, 0.0, -0.5)
    all_slots = np.concatenate([np.asarray(s) for s, _ in finished])
    assert len(set(all_slots.tolist())) == len(all_slots)  # slots unique
    probe = all_slots[:32]
    obs = {k: store.obs[k][probe] for k in store.obs}
    with torch.no_grad():
        want = masked_log_softmax(ref(to_torch(obs, torch.device("cpu")))).numpy()
    assert np.allclose(store.ref_lp[probe].astype(np.float32), want, atol=1e-2)
    n_games = st["games"]
    finished2, st2 = col.collect(200)  # continues without reset
    assert sum(len(s) for s, _ in finished2) >= 200
    assert st2["games"] > 0 and st2["games"] + n_games > n_games
    both = np.concatenate([all_slots, np.concatenate([np.asarray(s) for s, _ in finished2])])
    assert len(set(both.tolist())) == len(both)


def test_bc_seat_decisions_never_stored(engine):
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    col, store, _ = _collector(engine, decks, 1.0, seed=1)
    col.vs_bc[:] = True
    col.lseat[:] = 0  # learner seat known; every game redraws vs_bc=True (p_bc=1.0)
    finished, st = col.collect(500)
    total = sum(len(s) for s, _ in finished)
    assert total >= 500
    # only the learner seat is stored: finished samples are ~half of all decisions sent
    assert 0.2 < total / st["decisions"] < 0.75


def test_overlap_pipeline_three_iterations(engine):
    """train_rl's overlap building blocks, 3 iterations on the tiny VecEnv config above:
    background collect + GAE + PPO update, actor synced to the model after each update."""
    tr = _train_rl_mod()
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    dev = torch.device("cpu")
    model, ref = _tiny(decks, True), _tiny(decks, False)
    actor = copy.deepcopy(model).eval()
    vec = VecEnv(engine, decks, 8, 2, seed=0)
    vec.set_max_decisions(60)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False))
    rng = np.random.default_rng(0)
    col = Collector(vec, actor, ref, store, 0, False, dev, 0.3, -0.5, 2,
                    np.random.default_rng(1234), "fp32")
    trunk = [p for n, p in model.named_parameters() if not n.startswith("value_head.")]
    opt = torch.optim.AdamW([{"params": trunk}, {"params": list(model.value_head.parameters()), "lr": 1e-3}],
                            lr=1e-3, weight_decay=0.0)
    vopt = torch.optim.AdamW(model.value_head.parameters(), lr=1e-3)
    recs = []
    t0 = time.time()
    pending = col.collect(200)
    for it in (1, 2, 3):
        task = tr.CollectTask(col, 200).start()
        slots, adv, ret, adv_n = tr.gae_batch(store, pending[0], 0.95)
        assert len(slots) > 0
        agg, t_update = tr.ppo_update(
            model, model, opt, vopt, store, slots, adv_n, ret, rng, dev,
            clip=0.2, vf_coef=0.5, kl_coef=0.1, ent_coef=0.003, epochs=1,
            minibatch=64, warm=(it == 1), precision="fp32", max_upload_gb=6.0, compile=False)
        assert agg and all(np.isfinite(v) for v in agg.values())
        finished_next, _ = task.join()
        for s, _ in pending[0]:
            store.release(s)
        tr.sync_actor(actor, model)
        for k, v in model.state_dict().items():
            assert torch.equal(actor.state_dict()[k].cpu(), v.cpu()), k
        pending = (finished_next, None)
        recs.append({"iter": it, "overlap": True, "samples": len(slots),
                     **{k: float(v) for k, v in agg.items()}})
    assert [r["iter"] for r in recs] == [1, 2, 3]
    assert all(r["overlap"] is True for r in recs)
    assert all(np.isfinite(r["samples"]) and r["samples"] > 0 for r in recs)
    assert time.time() - t0 < 300


def test_overlap_thread_exception_propagates():
    """A collect that raises on its 2nd call surfaces in the joining thread within seconds."""
    tr = _train_rl_mod()
    calls = {"n": 0}

    class _Col:
        def collect(self, n):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom-2nd-collect")
            return [([0], 1.0)], {"games": 1}

    col = _Col()
    first = tr.CollectTask(col, 10).start()
    assert first.join()[0] == [([0], 1.0)]
    bad = tr.CollectTask(col, 10).start()
    t0 = time.time()
    try:
        bad.join(timeout=30)
    except RuntimeError as e:
        assert "boom-2nd-collect" in str(e)
    else:
        raise AssertionError("collect exception did not propagate")
    assert time.time() - t0 < 30


def test_hint_step_stats_are_not_minibatch_averaged(engine):
    """The hint step reports its own CE/accuracy (one step), and the PPO stats are identical with or without it."""
    tr = _train_rl_mod()
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    dev = torch.device("cpu")
    model, ref = _tiny(decks, True), _tiny(decks, False)
    vec = VecEnv(engine, decks, 8, 2, seed=0)
    vec.set_max_decisions(60)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False))
    col = Collector(vec, copy.deepcopy(model).eval(), ref, store, 0, False, dev, 0.3, -0.5, 2,
                    np.random.default_rng(1234), "fp32")
    finished, _ = col.collect(200)
    slots, _adv, ret, adv_n = tr.gae_batch(store, finished, 0.95)
    hobs = {k: v[np.asarray(slots)[:32]] for k, v in store.obs.items()}
    mask = hobs["action_mask"]
    hint = {"obs": hobs, "act": np.argmax(mask, 1).astype(np.int32), "coef": 1.0, "batch": 32}
    kw = dict(clip=0.2, vf_coef=0.5, kl_coef=0.1, ent_coef=0.003, epochs=1, minibatch=64, warm=False,
              precision="fp32", max_upload_gb=6.0, compile=False)
    m0 = copy.deepcopy(model)
    opt0 = torch.optim.AdamW(m0.parameters(), lr=1e-3)
    a0, _ = tr.ppo_update(m0, m0, opt0, torch.optim.AdamW(m0.value_head.parameters(), lr=1e-3), store, slots,
                          adv_n, ret, np.random.default_rng(5), dev, **kw)
    m1 = copy.deepcopy(model)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    a1, _ = tr.ppo_update(m1, m1, opt1, torch.optim.AdamW(m1.value_head.parameters(), lr=1e-3), store, slots,
                          adv_n, ret, np.random.default_rng(5), dev, hint=hint, **kw)
    assert {k: v for k, v in a1.items() if not k.startswith("hint_")} == a0
    assert 0.0 <= a1["hint_acc"] <= 1.0 and a1["hint_ce"] > 0.05  # an untrained tiny net is not near-zero CE


def _collector_log(engine, decks, seed=0, focus_idx=None, log_games=True):
    dev = torch.device("cpu")
    model, ref = _tiny(decks, True), _tiny(decks, False)
    vec = VecEnv(engine, decks, 8, 2, seed=seed)
    vec.set_max_decisions(60)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False))
    col = Collector(vec, model, ref, store, 0, False, dev, 0.3, -0.5, 2,
                    np.random.default_rng(seed), "fp32",
                    focus_idx=focus_idx, log_games=log_games)
    return col, vec, decks


def test_dtype_entropy_and_summarize_lines_handmade():
    from decklight.rl.rollout import dtype_entropy, summarize_lines
    import math
    assert dtype_entropy({}) == 0.0
    assert dtype_entropy({3: 5}) == 0.0
    assert dtype_entropy({1: 2, 2: 2}) == pytest.approx(math.log(2))
    assert summarize_lines({}) == (0, 0.0)
    assert summarize_lines({((1, 7),): 3, ((1, 8),): 1}) == (2, 0.75)


def test_log_games_records_replay_in_reference_env(engine):
    """Logged records reproduce: DuelConfig(decks, seed) + choices replays the same
    choices, and the winner for non-truncated games (truncated ones continue)."""
    from decklight.engine.duel import DuelConfig
    from decklight.engine.env import PyEnv
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    col, _, _ = _collector_log(engine, decks)
    _, st = col.collect(300)
    assert st["games"] > 0 and len(st["games_log"]) == st["games"]
    for g in st["games_log"][:3]:
        assert g["deck0"] >= 0 and g["deck1"] >= 0 and g["choices"] and len(g["choices"]) == len(g["actors"])
        assert g["opponent"] in ("self", "bc") and g["learner_seat"] in (0, 1)
        cfg = DuelConfig(decks[g["deck0"]], decks[g["deck1"]], seed=g["seed"])
        py = PyEnv.replay(engine, cfg, g["choices"])
        assert list(py.choices) == g["choices"]  # the deal + choices reproduce the game
        assert len(py.choices) == len(g["choices"])
        if not g["truncated"]:
            assert py.done and py.winner == g["winner"]


def test_log_games_off_keeps_stats_lean(engine):
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    col, _, _ = _collector_log(engine, decks, log_games=False)
    _, st = col.collect(200)
    assert st["games_log"] == [] and st["games"] > 0


def test_focus_diversity_stats_on_tiny_run(engine):
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    col, _, _ = _collector_log(engine, decks, focus_idx={0, 1})
    _, st = col.collect(300)
    assert st["focus_dtype_n"] > 0 and st["focus_dtype_entropy"] >= 0.0
    assert st["focus_line_distinct"] <= st["focus_line_n"] or st["focus_line_n"] == 0
    if st["focus_line_n"]:
        assert 0.0 <= st["focus_line_top_share"] <= 1.0
        assert st["focus_line_top"] <= st["focus_line_n"]
    else:
        assert st["focus_line_top_share"] == 0.0


def test_non_finite_minibatch_step_is_skipped(engine, capsys):
    """A minibatch with a non-finite gradient skips its optimizer step (weights stay finite, ALERT printed); non-finite
    gradients in > 5% of minibatches raise (stage P3 smoke 1, NOTEBOOK 2026-10-20)."""
    tr = _train_rl_mod()
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    dev = torch.device("cpu")
    model, ref = _tiny(decks, True), _tiny(decks, False)
    vec = VecEnv(engine, decks, 8, 2, seed=0)
    vec.set_max_decisions(60)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False))
    col = Collector(vec, copy.deepcopy(model).eval(), ref, store, 0, False, dev, 0.3, -0.5, 2,
                    np.random.default_rng(1234), "fp32")
    finished, _ = col.collect(200)
    slots, _adv, ret, adv_n = tr.gae_batch(store, finished, 0.95)
    kw = dict(clip=0.2, vf_coef=0.5, kl_coef=0.1, ent_coef=0.003, epochs=1, minibatch=64, warm=False,
              precision="fp32", max_upload_gb=6.0, compile=False)

    def run(poison):
        m = copy.deepcopy(model)
        calls = [0]

        def hook(g):
            calls[0] += 1
            return torch.full_like(g, float("nan")) if poison(calls[0]) else g

        next(p for p in m.parameters() if p.requires_grad).register_hook(hook)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        tr.ppo_update(m, m, opt, torch.optim.AdamW(m.value_head.parameters(), lr=1e-3), store, slots, adv_n, ret,
                      np.random.default_rng(5), dev, **kw)
        return m, calls[0]

    m, n_mb = run(lambda c: c == 1)
    assert n_mb >= 2
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert "ALERT nan_skip: 1/" in capsys.readouterr().out
    try:
        run(lambda c: True)
    except RuntimeError as e:
        assert "non-finite gradients" in str(e)
    else:
        raise AssertionError("all-NaN minibatches did not raise")
