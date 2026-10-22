"""VRPO advantages (RL v2 step 2): Q head, expected-SARSA(λ) targets, tiny collect+update (CPU)."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from decklight.decks import load_ydk
from decklight.engine.cpp_env import VecEnv
from decklight.model.policy import PolicyNet
from decklight.rl.ppo import SlotStore, gather, masked_log_softmax, to_torch
from decklight.rl.rollout import Collector
from decklight.rl.vrpo import ppo_loss_vrpo, vbar, vrpo_trajectory

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"

_TRAIN_RL = None


def _train_rl_mod():
    global _TRAIN_RL
    if _TRAIN_RL is None:
        p = Path(__file__).resolve().parents[2] / "scripts" / "train_rl.py"
        spec = importlib.util.spec_from_file_location("train_rl_vrpo_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _TRAIN_RL = mod
    return _TRAIN_RL


def _decks(n=2):
    return [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))[:n]]


def _emb_for(decks):
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    return {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
            "effects": rng.normal(size=(n, 2, 8)).astype(np.float32),
            "n_effects": np.ones(n, np.int64)}  # fmt: skip


def _tiny(decks, known=False, value=False, q=False):
    return PolicyNet(_emb_for(decks), d=16, layers=1, heads=2, dropout=0.0,
                     known=known, value=value, q=q).eval()


def _batch(b=4, h=32, n_legal=2):
    mask = torch.zeros(b, 64, dtype=torch.bool)
    mask[:, :n_legal] = True
    return {
        "cards": torch.zeros(b, 160, 21),
        "card_mask": torch.ones(b, 160, dtype=torch.bool),
        "global": torch.zeros(b, 24),
        "chain": torch.zeros(b, 16, 5),
        "chain_mask": torch.zeros(b, 16, dtype=torch.bool),
        "actions": torch.zeros(b, 64, 10),
        "action_mask": mask,
        "history": torch.zeros(b, h, 13),
        "history_mask": torch.ones(b, h, dtype=torch.bool),
    }


def test_q_false_state_dict_and_outputs_unchanged():
    decks = _decks()
    a = _tiny(decks)  # q=False default
    assert not any(k.startswith("q_head.") for k in a.state_dict())
    assert not hasattr(a, "q_head")
    torch.manual_seed(0)
    b = _tiny(decks)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=True)  # old ckpts load strictly
    assert not missing and not unexpected
    obs = _batch()
    with torch.no_grad():
        assert torch.equal(a.eval()(obs), b.eval()(obs))
    try:
        a.forward_q(obs)
    except AttributeError:
        pass
    else:
        raise AssertionError("q=False net must have no forward_q")


def test_q_init_matches_value():
    decks = _decks()
    torch.manual_seed(1)
    net = _tiny(decks, value=True, q=True).eval()
    obs = _batch()
    with torch.no_grad():
        _logits, q = net.forward_q(obs)
        _, v = net.forward_value(obs)
    assert q.shape == (4, 64) and torch.isfinite(q).all()
    legal = obs["action_mask"]
    assert torch.allclose(q[legal], v.unsqueeze(1).expand_as(q)[legal]), "Q(legal) == V at init"
    assert (q[~legal] == 0).all(), "illegal Q is 0 (finite, masked out downstream)"


def test_vrpo_targets_match_hand_computation():
    q = np.array([0.1, 0.2, 0.3], np.float32)
    vb = np.array([0.0, 0.1, -0.1], np.float32)
    adv0, ret0 = vrpo_trajectory(q, vb, 1.0, lam=0.0)
    assert np.allclose(ret0, [0.1, -0.1, 1.0]), ret0  # G_t = Vbar(s_{t+1}), G_T = r
    assert np.allclose(adv0, q - vb), adv0
    adv1, ret1 = vrpo_trajectory(q, vb, 1.0, lam=1.0)
    want = np.array([0.5, 0.6, 1.0], np.float32)  # r + cumsum of (Vbar_k - Q_k) above t
    assert np.allclose(ret1, want), (ret1, want)
    assert np.allclose(adv1, q - vb)


def test_uniform_q_gives_zero_advantage_and_vbar_is_v():
    rng = np.random.default_rng(0)
    n_opt, b = 64, 3
    v = rng.normal(size=b).astype(np.float32)
    q = np.repeat(v[:, None], n_opt, axis=1)
    legal = np.zeros((b, n_opt), bool)
    legal[:, :5] = True
    pi = rng.random((b, n_opt)).astype(np.float32) * legal
    pi /= pi.sum(-1, keepdims=True)
    assert np.allclose(vbar(pi, q), v), "Vbar = V when Q(.,a) == V"
    adv, _ = vrpo_trajectory(q_taken=v.copy(), vbar_vals=v.copy(), reward=1.0, lam=0.95)
    assert np.allclose(adv, 0.0)


def test_illegal_options_never_nan():
    decks = _decks()
    net = _tiny(decks, value=True, q=True)
    obs = _batch(b=4, n_legal=3)
    logits, q = net.forward_q(obs)
    assert torch.isfinite(logits[obs["action_mask"]]).all() and torch.isfinite(q).all()
    pi = (masked_log_softmax(logits).exp() * torch.isfinite(logits)).detach().numpy()
    assert (pi[:, 3:] == 0).all() and np.isfinite(vbar(pi, q.detach().numpy())).all()
    adv, ret = vrpo_trajectory(np.array([0.1, -0.2, 0.0, 0.3], np.float32),
                               np.array([0.0, 0.1, -0.1, 0.2], np.float32), -1.0, lam=0.95)
    assert np.isfinite(adv).all() and np.isfinite(ret).all()
    act = torch.zeros(4, dtype=torch.long)
    loss, st = ppo_loss_vrpo(net, masked_log_softmax(logits).detach(), obs, act,
                             torch.full((4,), -0.69), torch.randn(4), torch.ones(4),
                             0.2, 0.5, 0.1, 0.01)
    loss.backward()
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in st.values()), st


def _collect_vrpo(engine, decks, priv_critic, n=300):
    dev = torch.device("cpu")
    policy = _tiny(decks, value=True, q=not priv_critic)
    ref = _tiny(decks)
    critic = _tiny(decks, known=True, value=True, q=True) if priv_critic else None
    vec = VecEnv(engine, decks, 8, 2, seed=0)
    vec.set_max_decisions(60)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False, priv=priv_critic), vrpo=True)
    assert store.pi is not None and store.q is not None
    col = Collector(vec, policy, ref, store, 0, False, dev, 0.3, -0.5, 2,
                    np.random.default_rng(0), "fp32", critic=critic,
                    priv_critic=priv_critic, adv="vrpo")
    finished, _st = col.collect(n)
    assert len(finished) > 0
    return policy, ref, critic, store, finished


def test_vrpo_collect_and_update_no_priv(engine):
    tr = _train_rl_mod()
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    dev = torch.device("cpu")
    policy, _ref, _, store, finished = _collect_vrpo(engine, decks, False)
    assert (store.pi is not None and store.q is not None)
    probe = np.concatenate([np.asarray(s) for s, _ in finished])
    assert (store.pi[probe].astype(np.float32).sum(-1) > 0.99).all(), "pi rows sum to 1 over legal"
    assert np.isfinite(store.q[probe].astype(np.float32)).all()
    slots, adv, ret, adv_n, info = tr.vrpo_targets_batch(store, finished, 0.95)
    assert len(slots) > 0 and np.isfinite(adv).all() and np.isfinite(ret).all()
    assert np.isfinite(info["vbar_mean"]) and np.isfinite(info["adv_diff"]) and info["adv_diff"] > 0, info
    before = {k: v.clone() for k, v in policy.q_head.state_dict().items()}
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    vopt = torch.optim.AdamW(policy.q_head.parameters(), lr=1e-3)
    agg, _ = tr.ppo_update(
        policy, policy, opt, vopt, store, slots, adv_n, ret, np.random.default_rng(1), dev,
        clip=0.2, vf_coef=0.5, kl_coef=0.1, ent_coef=0.003, epochs=1, minibatch=64,
        warm=False, precision="fp32", max_upload_gb=6.0, compile=False, adv="vrpo")
    assert "q_ev" in agg and all(np.isfinite(v) for v in agg.values()), agg
    assert any(not torch.equal(before[k], v) for k, v in policy.q_head.state_dict().items())
    # warm-up trains the Q head only: value head and trunk stay put
    v_before = {k: v.clone() for k, v in policy.value_head.state_dict().items()}
    agg_w, _ = tr.ppo_update(
        policy, policy, opt, vopt, store, slots, adv_n, ret, np.random.default_rng(2), dev,
        clip=0.2, vf_coef=0.5, kl_coef=0.1, ent_coef=0.003, epochs=1, minibatch=64,
        warm=True, precision="fp32", max_upload_gb=6.0, compile=False, adv="vrpo")
    assert np.isfinite(agg_w["q_ev"])
    assert all(torch.equal(v_before[k], v) for k, v in policy.value_head.state_dict().items())
    for s, _ in finished:
        store.release(s)


def test_vrpo_collect_and_update_priv(engine):
    tr = _train_rl_mod()
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    dev = torch.device("cpu")
    policy, _, critic, store, finished = _collect_vrpo(engine, decks, True)
    assert critic is not None
    slots, _adv, ret, adv_n, info = tr.vrpo_targets_batch(store, finished, 0.95)
    assert info["adv_diff"] > 0, info
    before_c = {k: v.clone() for k, v in critic.q_head.state_dict().items()}
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    vopt = torch.optim.AdamW(policy.value_head.parameters(), lr=1e-3)
    copt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    agg, _ = tr.ppo_update(
        policy, policy, opt, vopt, store, slots, adv_n, ret, np.random.default_rng(1), dev,
        clip=0.2, vf_coef=0.5, kl_coef=0.1, ent_coef=0.003, epochs=1, minibatch=64,
        warm=False, precision="fp32", max_upload_gb=6.0, compile=False,
        critic=critic, copt=copt, priv_critic=True, adv="vrpo")
    assert {"q_ev", "policy_ev"} <= set(agg) and all(np.isfinite(v) for v in agg.values()), agg
    assert any(not torch.equal(before_c[k], v) for k, v in critic.q_head.state_dict().items())
    # warm-up trains the critic Q head only
    before_t = {k: v.clone() for k, v in critic.state_dict().items() if not k.startswith("q_head.")}
    agg_w, _ = tr.ppo_update(
        policy, policy, opt, vopt, store, slots, adv_n, ret, np.random.default_rng(2), dev,
        clip=0.2, vf_coef=0.5, kl_coef=0.1, ent_coef=0.003, epochs=1, minibatch=64,
        warm=True, precision="fp32", max_upload_gb=6.0, compile=False,
        critic=critic, copt=copt, priv_critic=True, adv="vrpo")
    assert np.isfinite(agg_w["q_ev"])
    after_t = {k: v for k, v in critic.state_dict().items() if not k.startswith("q_head.")}
    assert all(torch.equal(before_t[k], after_t[k]) for k in before_t), "warm-up leaves trunk/value alone"
    for s, _ in finished:
        store.release(s)


def test_vrpo_without_q_head_fails_fast(engine):
    decks = _decks()
    dev = torch.device("cpu")
    policy, ref = _tiny(decks, value=True), _tiny(decks)
    vec = VecEnv(engine, decks, 8, 2, seed=0)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False), vrpo=True)
    try:
        Collector(vec, policy, ref, store, 0, False, dev, 0.3, -0.5, 2,
                  np.random.default_rng(0), "fp32", adv="vrpo")
    except ValueError:
        pass
    else:
        raise AssertionError("--adv vrpo without a Q head must raise")


def test_load_net_q_head_opt_in(tmp_path):
    tr = _train_rl_mod()
    decks = _decks()
    emb = _emb_for(decks)
    a = {"d": 16, "layers": 1, "heads": 2}
    bc = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0)
    ckpt = tmp_path / "bc.pt"
    torch.save(bc.state_dict(), ckpt)
    net = tr.load_net(ckpt, emb, a, value=True, q=True)
    assert net.has_q and hasattr(net, "q_head")
    for k, v in bc.state_dict().items():
        assert torch.equal(net.state_dict()[k], v), k
    with torch.no_grad():
        _, q = net.forward_q(to_torch({k: v for k, v in _batch().items()}, torch.device("cpu")))
        _, v = net.forward_value(to_torch({k: v for k, v in _batch().items()}, torch.device("cpu")))
    assert torch.isfinite(q).all()
    critic = tr.load_critic(ckpt, emb, a, q=True)
    assert critic.has_q


def test_q_head_hidden_layer_gets_gradient():
    """Review fix: zero-initializing both q_head layers left every weight with zero gradient (only the last bias moved)."""
    decks = _decks()
    torch.manual_seed(1)
    net = _tiny(decks, value=True, q=True).train()
    _logits, q = net.forward_q(_batch())
    (q - 1.0).pow(2).sum().backward()
    assert net.q_head[0].weight.grad is not None
    assert net.q_head[-1].weight.grad.abs().sum() > 0, "output layer weights get gradient"
