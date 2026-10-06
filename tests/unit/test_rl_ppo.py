import numpy as np
import torch

from ygo_zero.model.policy import PolicyNet
from ygo_zero.rl.ppo import SlotStore, gae, masked_log_softmax, ppo_loss


def _emb(d=8, n=10):
    rng = np.random.default_rng(0)
    return {
        "codes": np.arange(1, n + 1),
        "card": rng.normal(size=(n, d)).astype(np.float32),
        "effects": rng.normal(size=(n, 2, d)).astype(np.float32),
        "n_effects": np.ones(n, dtype=np.int64),
    }


def _batch(b=4, h=128):
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


def test_gae_terminal_reward_lambda1_is_monte_carlo():
    v = np.array([0.1, -0.2, 0.3], np.float32)
    adv, ret = gae(v, 1.0, lam=1.0)
    assert np.allclose(ret, 1.0) and np.allclose(adv, 1.0 - v)
    adv0, _ = gae(v, -1.0, lam=0.0)  # one-step TD
    assert np.allclose(adv0, [-0.2 - 0.1, 0.3 + 0.2, -1.0 - 0.3])


def test_masked_log_softmax_no_nan():
    lg = torch.tensor([[0.0, float("-inf"), 1.0]])
    lp = masked_log_softmax(lg)
    assert torch.isfinite(lp).all() and lp[0, 1] == 0 and torch.isclose(lp[0, [0, 2]].exp().sum(), torch.tensor(1.0))


def test_value_head_keeps_bc_loading_and_ppo_step_runs():
    emb = _emb()
    bc = PolicyNet(emb, d=16, layers=1, heads=2)
    net = PolicyNet(emb, d=16, layers=1, heads=2, value=True, dropout=0.0)
    missing, unexpected = net.load_state_dict(bc.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("value_head.") for k in missing)
    obs = _batch(b=4, h=32)
    logits, v = net.forward_value(obs)
    assert logits.shape == (4, 64) and v.shape == (4,) and (v.abs() <= 1).all()
    act = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        ref_lp = masked_log_softmax(bc.eval()(obs))
    loss, st = ppo_loss(net, ref_lp, obs, act, torch.full((4,), -0.69), torch.randn(4), torch.ones(4), 0.2, 0.5,
                        0.1, 0.01)  # fmt: skip
    loss.backward()
    assert torch.isfinite(loss) and st["kl_bc"] >= -1e-6
    assert net.value_head[1].weight.grad is not None


def test_cached_ref_kl_matches_module_path():
    emb = _emb()
    bc = PolicyNet(emb, d=16, layers=1, heads=2).eval()
    net = PolicyNet(emb, d=16, layers=1, heads=2, value=True, dropout=0.0).eval()
    obs = _batch(b=4, h=32)
    act = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        cached = masked_log_softmax(bc(obs))
    _, st = ppo_loss(net, cached, obs, act, torch.full((4,), -0.69), torch.randn(4), torch.ones(4), 0.2, 0.5,
                     0.1, 0.01)  # fmt: skip
    with torch.no_grad():  # the old module path, computed inline
        logits, _ = net.forward_value(obs)
        lp_all = masked_log_softmax(logits)
        p = lp_all.exp() * torch.isfinite(logits)
        old_kl = (p * (lp_all - masked_log_softmax(bc(obs)))).sum(1).mean().item()
    assert abs(st["kl_bc"] - old_kl) <= 1e-5


def test_slot_store_put_release():
    st = SlotStore(5, {"x": np.zeros((1, 3), np.int32)})
    s = st.put({"x": np.ones((2, 3), np.int32)}, np.array([1, 2]), np.zeros(2), np.zeros(2),
               np.full((2, 64), -0.5, np.float32))
    assert st.free == 3 and (st.obs["x"][s] == 1).all()
    assert st.ref_lp.shape == (5, 64) and st.ref_lp.dtype == np.float16
    assert np.allclose(st.ref_lp[s].astype(np.float32), -0.5, atol=1e-2)
    st.release(s)
    assert st.free == 5
