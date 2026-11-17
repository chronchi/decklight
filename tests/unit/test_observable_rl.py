"""PPO on an observable-history net (X2, NOTEBOOK 2026-11-10): the rollout store and gather carry obs_history."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from decklight.decks import load_ydk
from decklight.engine.cpp_env import VecEnv
from decklight.model.policy import PolicyNet
from decklight.rl.ppo import SlotStore, gather, model_keys, to_torch
from decklight.rl.remote import EpisodeLog, read_episode

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"


def _tiny(decks, observable):
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    torch.manual_seed(0)
    return PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, value=True, observable=observable).eval()


def test_model_keys_observable():
    assert "obs_history" not in model_keys(0, False)
    k = model_keys(0, False, observable=True)
    assert {"obs_history", "obs_history_mask", "history", "history_mask"} <= set(k)


def test_observable_net_reads_obs_history_from_vec_buffer(engine):
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))[:2]]
    vec = VecEnv(engine, decks, 4, 2, seed=5)
    vec.reset()
    for _ in range(6):  # a few decisions so the two history windows differ
        vec.step(np.array([int(np.flatnonzero(vec.buf["action_mask"][i])[0]) for i in range(4)]))
    rows = np.arange(4)
    full, obs = gather(vec.buf, rows, 0, False), gather(vec.buf, rows, 0, False, observable=True)
    assert set(obs) - set(full) == {"obs_history", "obs_history_mask"}
    assert not np.array_equal(obs["obs_history"], obs["history"]), "windows must differ for the check to bite"
    net, plain = _tiny(decks, True), _tiny(decks, False)
    plain.load_state_dict({k: v for k, v in net.state_dict().items() if k != "hist_observable"})
    with pytest.raises(KeyError):
        net(to_torch(full, torch.device("cpu")))
    swapped = {**obs, "history": obs["obs_history"], "history_mask": obs["obs_history_mask"]}
    with torch.no_grad():
        got = net(to_torch(obs, torch.device("cpu")))
        want = plain(to_torch({k: v for k, v in swapped.items() if not k.startswith("obs_")}, torch.device("cpu")))
    assert torch.equal(got, want)
    store = SlotStore(16, gather(vec.buf, np.arange(1), 0, False, observable=True))
    assert "obs_history" in store.obs


def test_episode_log_keeps_obs_history(tmp_path):
    log = EpisodeLog()
    ob = {"cards": np.zeros((160, 21), np.int32), "card_mask": np.zeros(160, bool), "global": np.zeros(32, np.int32),
          "chain": np.zeros((16, 8), np.int32), "chain_mask": np.zeros(16, bool),
          "actions": np.zeros((64, 12), np.int32), "action_mask": np.ones(64, bool),
          "history": np.zeros((32, 6), np.int32), "history_mask": np.zeros(32, bool),
          "obs_history": np.ones((32, 6), np.int32), "obs_history_mask": np.ones(32, bool)}  # fmt: skip
    log.add(ob, 0, -0.1, 0.0, 1)
    e = read_episode(log.write(tmp_path, 1.0, {}))
    assert (e["obs"]["obs_history"] == 1).all() and e["obs"]["obs_history_mask"].all()


def test_derived_run_args_keep_observable(tmp_path):
    """train_rl's metrics.jsonl copies ARCH_KEYS from --init: its own iter-*.pt must load as observable (BCPolicy)."""
    from decklight.eval.bc_policy import _run_args
    from decklight.model.policy_v3 import ARCH_KEYS

    a = {"d": 16, "layers": 1, "heads": 2, "hist": 0, "known": False, "observable_history": True}
    (tmp_path / "metrics.jsonl").write_text('{"args": %s}\n' % json.dumps({k: a.get(k) for k in ARCH_KEYS}))
    assert _run_args(tmp_path / "iter-0000.pt")["observable_history"] is True


def test_value_net_reads_obs_history(engine, tmp_path):
    """Value v2 (prereg 2026-11-11): a value head on Y's observable trunk scores search leaves from obs_history."""
    from decklight.eval.value import ValueNet

    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))[:2]]
    vec = VecEnv(engine, decks, 4, 2, seed=5)
    vec.reset()
    for _ in range(6):
        vec.step(np.array([int(np.flatnonzero(vec.buf["action_mask"][i])[0]) for i in range(4)]))
    obs = gather(vec.buf, np.arange(4), 0, False, observable=True)
    net = _tiny(decks, True)
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    np.savez(tmp_path / "emb.npz", codes=np.array(codes), card=rng.normal(size=(n, 8)).astype(np.float32),
             effects=rng.normal(size=(n, 2, 8)).astype(np.float32), n_effects=np.ones(n, np.int64))  # fmt: skip
    torch.save(net.state_dict(), tmp_path / "best.pt")
    args = {"d": 16, "layers": 1, "heads": 2, "dropout": 0.0, "value": True, "observable_history": True}
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"args": args}) + "\n")
    vn = ValueNet(tmp_path / "best.pt", tmp_path / "emb.npz")
    v = vn.values(obs)
    blank = {**obs, "obs_history_mask": np.zeros_like(obs["obs_history_mask"])}
    assert np.isfinite(v).all() and not np.allclose(v, vn.values(blank)), "the value must read obs_history"
