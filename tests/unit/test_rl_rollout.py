"""Collector (async rollout) over tiny PolicyNets on the fixture decks (CPU, fp32)."""

import numpy as np
import torch

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import VecEnv
from ygo_zero.model.policy import PolicyNet
from ygo_zero.rl.ppo import SlotStore, gather, masked_log_softmax, to_torch
from ygo_zero.rl.rollout import Collector

from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"


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
