"""Opponent-turn shaping/exploration: GAE step rewards, target windows, mu mixing, loss ISW."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from decklight.engine.actions import Kind
from decklight.model.policy import PolicyNet
from decklight.rl.ppo import gae, masked_log_softmax, ppo_loss
from decklight.rl.rollout import apply_isw_floor, sample_target_mixture, target_window_mask

_TRAIN_RL = None


def _train_rl_mod():
    global _TRAIN_RL
    if _TRAIN_RL is None:
        p = Path(__file__).resolve().parents[2] / "scripts" / "train_rl.py"
        spec = importlib.util.spec_from_file_location("train_rl_isw_floor_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _TRAIN_RL = mod
    return _TRAIN_RL


def _gae_today(values, reward, lam):
    n = len(values)
    nxt = np.append(values[1:], 0.0).astype(np.float32)
    delta = nxt - values
    delta[-1] += reward
    adv = np.zeros(n, np.float32)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        acc = delta[t] + lam * acc
        adv[t] = acc
    return adv, adv + values


def test_gae_step_rewards_hand_example_and_telescoping():
    v = np.array([0.1, -0.2, 0.3], np.float32)
    step = np.array([0.05, -0.1, 0.02], np.float32)
    adv, ret = gae(v, 1.0, 0.95, step_rewards=step)
    delta = np.array([v[1] - v[0], v[2] - v[1], -v[2] + 1.0], np.float32) + step
    want_adv = np.zeros(3, np.float32)
    acc = 0.0
    for t in range(2, -1, -1):
        acc = delta[t] + 0.95 * acc
        want_adv[t] = acc
    assert np.allclose(adv, want_adv) and np.allclose(ret, want_adv + v)

    phi = np.array([-0.1, -0.3, -0.2], np.float32)
    shaped = np.empty_like(phi)
    shaped[:-1] = phi[1:] - phi[:-1]
    shaped[-1] = -phi[-1]  # Phi(terminal) = 0
    _, shaped_ret = gae(v, 1.0, 1.0, step_rewards=shaped)
    assert np.isclose(shaped_ret[0], 1.0 - phi[0])  # terminal reward - Phi(s_0)


def test_gae_no_step_rewards_is_todays_path():
    v = np.array([0.1, -0.2, 0.3], np.float32)
    for lam in (0.0, 0.95, 1.0):
        a0, r0 = _gae_today(v, 1.0, lam)
        a1, r1 = gae(v, 1.0, lam)
        assert np.array_equal(a0, a1) and np.array_equal(r0, r1)


def test_target_window_predicate():
    g = np.zeros((4, 24), np.int32)
    a = np.zeros((4, 64, 10), np.int32)
    legal = np.zeros((4, 64), bool)
    g[0, 5] = 1  # own turn
    a[0, 0, 0], legal[0, 0] = int(Kind.ACTIVATE), True
    g[1, 20] = 1  # chain length 1
    a[1, 0, 0], legal[1, 0] = int(Kind.ACTIVATE), True
    a[2, 0, 0], legal[2, 0] = int(Kind.SUMMON), True  # no activation
    a[3, 0, 0], legal[3, 0] = int(Kind.ACTIVATE), True  # target
    assert target_window_mask(g, a, legal).tolist() == [False, False, False, True]


def test_mu_mixing_distribution_and_isw():
    pi = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    lp = torch.log(pi).reshape(1, 4)
    legal = torch.ones(1, 4, dtype=torch.bool)
    activate = torch.tensor([[True, True, False, False]])
    n = 20_000
    torch.manual_seed(0)
    a, lp_a, isw = sample_target_mixture(lp.repeat(n, 1), legal.repeat(n, 1),
                                        activate.repeat(n, 1), 0.25)
    mu = 0.75 * pi + 0.25 * torch.tensor([0.5, 0.5, 0.0, 0.0])
    counts = np.bincount(a, minlength=4) / n
    assert np.allclose(counts, mu.numpy(), atol=0.01)
    assert np.allclose(lp_a, np.log(pi.numpy())[a], atol=1e-6)
    assert np.allclose(isw, torch.clamp(pi / mu, max=1.0).numpy()[a], atol=1e-6)


def test_eps_zero_is_bit_identical_to_sample_from_lp():
    from decklight.rl.rollout import _sample_from_lp

    torch.manual_seed(123)
    lp = torch.log_softmax(torch.randn(8, 6), -1)
    legal = torch.rand(8, 6) > 0.3
    legal[:, 0] = True
    activate = legal & (torch.rand(8, 6) > 0.5)

    torch.manual_seed(999)
    a0, lp0 = _sample_from_lp(lp, legal)
    torch.manual_seed(999)
    a1, lp1, isw = sample_target_mixture(lp, legal, activate, 0.0)
    assert np.array_equal(a0, a1) and np.array_equal(lp0, lp1)
    assert (isw == 1.0).all()


def test_isw_floor_zero_is_bit_identical_to_no_floor():
    rng = np.random.default_rng(0)
    isw = rng.random(1000).astype(np.float32)
    stored, applied = apply_isw_floor(isw, 0.0)
    assert np.array_equal(stored, isw) and stored.dtype == isw.dtype
    assert not applied.any()


def test_isw_floor_applies_only_to_window_rows_and_never_above_one():
    g = np.zeros((4, 24), np.int32)
    a = np.zeros((4, 64, 10), np.int32)
    legal = np.zeros((4, 64), bool)
    g[0, 5] = 1  # own turn
    a[0, 0, 0], legal[0, 0] = int(Kind.ACTIVATE), True
    g[1, 20] = 1  # chain length 1
    a[1, 0, 0], legal[1, 0] = int(Kind.ACTIVATE), True
    a[2, 0, 0], legal[2, 0] = int(Kind.SUMMON), True  # no activation
    a[3, 0, 0], legal[3, 0] = int(Kind.ACTIVATE), True  # target window
    mask = target_window_mask(g, a, legal)
    assert mask.tolist() == [False, False, False, True]

    isw = np.ones(4, dtype=np.float32)
    isw[3] = 0.1
    isw[mask], applied = apply_isw_floor(isw[mask], 0.7)
    assert np.allclose(isw, [1.0, 1.0, 1.0, 0.7])
    assert applied.tolist() == [True]

    floored, _ = apply_isw_floor(np.array([0.0, 0.25, 0.5, 1.0], np.float32), 1.0)
    assert np.all(floored <= 1.0) and np.allclose(floored, 1.0)


def _opp_ns(**kw):
    base = {"win_explore": 0.1, "isw_floor": 0.0, "shape_opp": 0.0, "adv": "gae"}
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("kw", [
    {"isw_floor": -0.1},
    {"isw_floor": 1.1},
    {"win_explore": 0.0, "isw_floor": 0.5},
])
def test_isw_floor_validation_errors(kw):
    mod = _train_rl_mod()
    with pytest.raises(SystemExit):
        mod.validate_opp_shaping(_opp_ns(**kw), argparse.ArgumentParser())


def test_isw_floor_validation_accepts_valid_combos():
    mod = _train_rl_mod()
    ap = argparse.ArgumentParser()
    mod.validate_opp_shaping(_opp_ns(), ap)
    mod.validate_opp_shaping(_opp_ns(isw_floor=1.0), ap)


def _emb(d=8, n=10):
    rng = np.random.default_rng(0)
    return {
        "codes": np.arange(1, n + 1),
        "card": rng.normal(size=(n, d)).astype(np.float32),
        "effects": rng.normal(size=(n, 2, d)).astype(np.float32),
        "n_effects": np.ones(n, dtype=np.int64),
    }


def _batch(b=4, h=32):
    return {
        "cards": torch.zeros(b, 160, 21),
        "card_mask": torch.ones(b, 160, dtype=torch.bool),
        "global": torch.zeros(b, 24),
        "chain": torch.zeros(b, 16, 5),
        "chain_mask": torch.zeros(b, 16, dtype=torch.bool),
        "actions": torch.zeros(b, 64, 10),
        "action_mask": torch.zeros(b, 64, dtype=torch.bool).scatter_(1, torch.tensor([[0, 1]] * b), True),
        "history": torch.zeros(b, h, 13),
        "history_mask": torch.ones(b, h, dtype=torch.bool),
    }


def test_loss_with_isw_one_equals_today_exactly():
    net = PolicyNet(_emb(), d=16, layers=1, heads=2, value=True, dropout=0.0).eval()
    obs = _batch()
    act = torch.tensor([0, 1, 0, 1])
    olp = torch.full((4,), -0.69)
    adv = torch.randn(4)
    ret = torch.ones(4)
    with torch.no_grad():
        ref_lp = masked_log_softmax(net(obs))
    loss0, st0 = ppo_loss(net, ref_lp, obs, act, olp, adv, ret, 0.2, 0.5, 0.1, 0.01)
    loss1, st1 = ppo_loss(net, ref_lp, obs, act, olp, adv, ret, 0.2, 0.5, 0.1, 0.01,
                          isw=torch.ones(4))
    assert torch.equal(loss0, loss1)
    assert st0.keys() == st1.keys()
    assert all(torch.equal(st0[k], st1[k]) for k in st0)


def test_window_rows_matches_target_window_mask():
    from decklight.rl.ppo import window_rows

    g = np.zeros((4, 24), np.int32)
    a = np.zeros((4, 64, 10), np.int32)
    legal = np.zeros((4, 64), bool)
    g[0, 5] = 1
    a[0, 0, 0], legal[0, 0] = int(Kind.ACTIVATE), True
    g[1, 20] = 1
    a[1, 0, 0], legal[1, 0] = int(Kind.ACTIVATE), True
    a[2, 0, 0], legal[2, 0] = int(Kind.SUMMON), True
    a[3, 0, 0], legal[3, 0] = int(Kind.ACTIVATE), True
    got = window_rows({"global": torch.from_numpy(g), "actions": torch.from_numpy(a)}, torch.from_numpy(legal))
    assert got.tolist() == target_window_mask(g, a, legal).tolist() == [False, False, False, True]


def _kl_setup():
    torch.manual_seed(0)
    net = PolicyNet(_emb(), d=16, layers=1, heads=2, value=True, dropout=0.0).eval()
    obs = _batch()
    obs["actions"][:, 0, 0] = int(Kind.ACTIVATE)  # option 0 is an activation on every row
    obs["global"][:2, 5] = 1  # rows 0, 1: own turn (not windows); rows 2, 3: opponent-turn chain-0 windows
    act, olp, adv, ret = torch.tensor([0, 1, 0, 1]), torch.full((4,), -0.69), torch.randn(4), torch.ones(4)
    ref = torch.log(torch.tensor([[0.9, 0.1] + [1e-6] * 62] * 4))  # far from the net (KL > 0); illegal: p = 0
    return net, obs, act, olp, adv, ret, ref


def test_kl_win_mult_one_is_todays_loss_exactly():
    net, obs, act, olp, adv, ret, ref = _kl_setup()
    l0, s0 = ppo_loss(net, ref, obs, act, olp, adv, ret, 0.2, 0.5, 0.1, 0.01)
    l1, s1 = ppo_loss(net, ref, obs, act, olp, adv, ret, 0.2, 0.5, 0.1, 0.01, kl_win_mult=1.0)
    assert torch.equal(l0, l1) and all(torch.equal(s0[k], s1[k]) for k in s0)


@pytest.mark.parametrize("keep", [None, torch.tensor([True, True, True, True])])
def test_kl_win_mult_zero_drops_the_window_rows_kl_only(keep):
    net, obs, act, olp, adv, ret, ref = _kl_setup()
    _, s_all = ppo_loss(net, ref, obs, act, olp, adv, ret, 0.2, 0.5, 0.1, 0.01, keep=keep)
    _, s_zero = ppo_loss(net, ref, obs, act, olp, adv, ret, 0.2, 0.5, 0.1, 0.01, keep=keep, kl_win_mult=0.0)
    _, s_half = ppo_loss(net, ref, obs, act, olp, adv, ret, 0.2, 0.5, 0.1, 0.01, keep=keep, kl_win_mult=0.5)
    with torch.no_grad():
        logits = net(obs)
        lp, legal = masked_log_softmax(logits), torch.isfinite(logits)
        row = torch.where(legal, lp.exp() * (lp - ref), 0.0).sum(1)
    assert torch.allclose(s_all["kl_bc"], row.mean(), atol=1e-6)
    assert torch.allclose(s_zero["kl_bc"], row[:2].sum() / 4, atol=1e-6)  # the 2 window rows contribute 0
    assert torch.allclose(s_half["kl_bc"], (row[:2].sum() + 0.5 * row[2:].sum()) / 4, atol=1e-6)


@pytest.mark.parametrize("kw", [
    {"kl_win_mult": -1.0},
    {"kl_win_mult": 0.0, "priv_critic": 1},
    {"win_explore_lines_only": 1, "win_explore": 0.0, "line_starts": "x.jsonl"},
    {"win_explore_lines_only": 1, "win_explore": 0.25, "line_starts": None},
])
def test_kl_win_mult_and_lines_only_validation(kw):
    with pytest.raises(SystemExit):
        _train_rl_mod().validate_opp_shaping(_opp_ns(**kw), argparse.ArgumentParser())


def test_kl_win_mult_and_lines_only_accepts_valid():
    _train_rl_mod().validate_opp_shaping(
        _opp_ns(kl_win_mult=0.0, win_explore=0.25, win_explore_lines_only=1, line_starts="x.jsonl"),
        argparse.ArgumentParser())
