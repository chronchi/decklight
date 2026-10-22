"""Privileged critic (RL v2 part B): priv_code engine key, critic obs, tiny collect+update (CPU)."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from decklight.decks import load_ydk
from decklight.engine import constants as C
from decklight.engine.cpp_env import CppEnv, VecEnv
from decklight.engine.duel import DuelConfig
from decklight.model.policy import PolicyNet
from decklight.rl.critic import critic_obs
from decklight.rl.ppo import SlotStore, gather, to_torch
from decklight.rl.rollout import Collector

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"
LOC_OF_ID = {1: C.LOCATION_DECK, 2: C.LOCATION_HAND, 3: C.LOCATION_MZONE, 4: C.LOCATION_SZONE,
             5: C.LOCATION_GRAVE, 6: C.LOCATION_REMOVED, 7: C.LOCATION_EXTRA}  # fmt: skip

_TRAIN_RL = None


def _train_rl_mod():
    global _TRAIN_RL
    if _TRAIN_RL is None:
        p = Path(__file__).resolve().parents[2] / "scripts" / "train_rl.py"
        spec = importlib.util.spec_from_file_location("train_rl_priv_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _TRAIN_RL = mod
    return _TRAIN_RL


def _decks(n=2):
    return [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))[:n]]


def _tiny(decks, known=False, value=False):
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    return PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, known=known, value=value).eval()


def _check_priv_codes(env, obs):
    """priv_code is non-zero exactly on valid rows whose code column is 0, holding true codes."""
    m = obs["card_mask"].astype(bool)
    cards, pc = obs["cards"], obs["priv_code"]
    assert (pc[~m] == 0).all(), "padding rows must carry no privileged codes"
    assert ((pc[m] > 0) == (cards[m, 0] == 0)).all()
    me = int(env.player)
    dbg = env.cards()
    true = {(int(p), int(loc), int(s)): int(c)
            for p, loc, s, c in zip(dbg["player"], dbg["location"], dbg["sequence"], dbg["code"])}
    n_hidden = 0
    for i in np.flatnonzero(m):
        loc, seq1, rel = int(cards[i, 1]), int(cards[i, 2]), int(cards[i, 3])
        if loc in (1, 8):  # own deck (seq col 0, sorted by code) / overlay units: always visible
            assert pc[i] == 0
            continue
        key = (rel ^ me, LOC_OF_ID[loc], seq1 - 1)
        assert key in true, f"row {i}: no debug card for {key}"
        if cards[i, 0] == 0:
            assert pc[i] == true[key] != 0, f"row {i}: priv {pc[i]} != true {true[key]}"
            n_hidden += 1
        else:
            assert pc[i] == 0
    return n_hidden


def test_priv_code_marks_hidden_rows_with_true_codes(engine):
    decks = _decks()
    env = CppEnv(engine, DuelConfig(decks[0], decks[1], seed=7))
    rng = np.random.default_rng(0)
    obs = env.reset()
    n_hidden = 0
    for _ in range(25):
        assert obs is not None
        n_hidden += _check_priv_codes(env, obs)
        legal = np.flatnonzero(obs["action_mask"])
        assert len(legal) > 0
        obs = env.step(int(rng.choice(legal)))
        if obs is None:  # game ended: keep checking the next one
            obs = env.reset()
    assert n_hidden > 0, "a 25-step game must hide some cards"


def test_critic_obs_reveals_hidden_hand(engine):
    decks = _decks()
    env = CppEnv(engine, DuelConfig(decks[0], decks[1], seed=11))
    obs = env.reset()
    buf = {k: v[None] for k, v in obs.items()}
    rows = np.array([0])
    pol = gather(buf, rows, 0, False)
    assert "priv_code" not in pol, "the policy's gathered obs must not carry priv_code"
    pol_k = gather(buf, rows, 0, True)
    assert "priv_code" not in pol_k
    priv = buf["priv_code"][rows]
    cob = critic_obs(pol_k, priv)
    base_ck, base_kc = pol_k["card_known"], pol_k["known_code"]
    assert (cob["card_known"] == np.where((priv > 0) & (base_ck == 0), 1, base_ck)).all()
    assert (cob["known_code"] == np.where((priv > 0) & (base_ck == 0), priv, base_kc)).all()
    hidden_hand = (pol_k["cards"][0, :, 1] == 2) & (pol_k["cards"][0, :, 3] == 1) & (pol_k["cards"][0, :, 0] == 0)
    assert hidden_hand.any(), "the opponent opening hand must be hidden at reset"
    assert (cob["card_known"][0][hidden_hand] == 1).all()
    assert (cob["known_code"][0][hidden_hand] > 0).all()


def test_gather_priv_opt_in():
    from decklight.rl.ppo import model_keys
    buf = {k: np.zeros((2, 160), np.int32) for k in model_keys(0, False) if k not in ("cards",)}
    buf["cards"] = np.zeros((2, 160, 21), np.int32)
    buf["card_known"] = np.zeros((2, 160), np.uint8)
    buf["known_code"] = np.zeros((2, 160), np.int32)
    buf["priv_code"] = np.arange(320, dtype=np.int32).reshape(2, 160)
    assert "priv_code" not in gather(buf, np.array([0]), 0, False)
    assert "priv_code" not in gather(buf, np.array([0]), 0, True)
    got = gather(buf, np.array([1]), 0, False, priv=True)
    assert (got["priv_code"] == np.arange(160, 320, dtype=np.int32)).all()


def test_policy_ignores_priv_code_critic_sees_it(engine):
    decks = _decks()
    dev = torch.device("cpu")
    policy, critic = _tiny(decks), _tiny(decks, known=True, value=True)
    vec = VecEnv(engine, decks, 4, 2, seed=3)
    vec.reset()
    rows = np.arange(4)
    pol = gather(vec.buf, rows, 0, False)
    priv = vec.buf["priv_code"][rows]
    assert (priv > 0).any(), "some cards must be hidden at reset"
    with torch.no_grad():
        tob = to_torch(pol, dev)
        l0 = policy(tob)
        l1 = policy({**tob, "priv_code": torch.from_numpy(np.ascontiguousarray(priv))})
    assert torch.equal(l0, l1), "extra priv_code key must not change policy logits"
    cob = critic_obs(pol, priv)
    assert (cob["card_known"][priv > 0].astype(bool)).any()
    with torch.no_grad():
        _, v1 = critic.forward_value(to_torch(cob, dev))
        blind = dict(cob)
        blind["card_known"] = np.zeros_like(cob["card_known"])
        blind["known_code"] = np.zeros_like(cob["known_code"])
        _, v0 = critic.forward_value(to_torch(blind, dev))
    assert not torch.allclose(v0, v1), "hiding the privileged cards must change critic values"


def test_priv_critic_collect_and_update(engine):
    tr = _train_rl_mod()
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    dev = torch.device("cpu")
    policy, ref = _tiny(decks, value=True), _tiny(decks)
    critic = _tiny(decks, known=True, value=True)
    vec = VecEnv(engine, decks, 8, 2, seed=0)
    vec.set_max_decisions(60)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False, priv=True))
    assert "priv_code" in store.obs
    col = Collector(vec, policy, ref, store, 0, False, dev, 0.3, -0.5, 2,
                    np.random.default_rng(0), "fp32", critic=critic, priv_critic=True)
    finished, st = col.collect(300)
    assert st["priv_rows"] > 0 and st["priv_nonzero"] > 0, st
    assert (store.obs["priv_code"][np.concatenate([np.asarray(s) for s, _ in finished])] > 0).any()
    slots, adv, ret, adv_n = tr.gae_batch(store, finished, 0.95)
    assert len(slots) > 0
    before_c = {k: v.clone() for k, v in critic.state_dict().items()}
    before_p = {k: v.clone() for k, v in policy.state_dict().items()}
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    copt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    vopt = torch.optim.AdamW(policy.value_head.parameters(), lr=1e-3)
    agg, _ = tr.ppo_update(
        policy, policy, opt, vopt, store, slots, adv_n, ret, np.random.default_rng(1), dev,
        clip=0.2, vf_coef=0.5, kl_coef=0.1, ent_coef=0.003, epochs=1, minibatch=64,
        warm=False, precision="fp32", max_upload_gb=6.0, compile=False,
        critic=critic, copt=copt, priv_critic=True)
    assert {"critic_ev", "policy_ev"} <= set(agg) and all(np.isfinite(v) for v in agg.values()), agg
    assert any(not torch.equal(before_c[k], v) for k, v in critic.state_dict().items())
    assert any(not torch.equal(before_p[k], v) for k, v in policy.state_dict().items())
    for s, _ in finished:
        store.release(s)


def test_load_critic_from_bc_checkpoint(tmp_path):
    tr = _train_rl_mod()
    decks = _decks()
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    bc = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0)  # BC: no known inputs, no value head
    ckpt = tmp_path / "bc.pt"
    torch.save(bc.state_dict(), ckpt)
    critic = tr.load_critic(ckpt, emb, {"d": 16, "layers": 1, "heads": 2})
    assert critic.known and hasattr(critic, "value_head")
    for k, v in bc.state_dict().items():  # trunk weights carry over exactly
        assert torch.equal(critic.state_dict()[k], v), k
