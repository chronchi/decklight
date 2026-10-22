import numpy as np
import torch

from decklight.model.policy import PolicyNet
from decklight.rl.ppo import SlotStore, gae, masked_log_softmax, ppo_loss


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


def test_device_accumulated_means_match_item_path():
    """New stats path (detached 0-d tensors, summed on-device, one .item() per key per
    iteration) gives the same per-iteration means as the old per-minibatch .item() path."""
    torch.manual_seed(7)
    emb = _emb()
    net = PolicyNet(emb, d=16, layers=1, heads=2, value=True, dropout=0.0)
    net.eval()
    old: dict[str, list[float]] = {}
    sums: dict[str, torch.Tensor] = {}
    n = 0
    for i in range(4):
        g = torch.Generator().manual_seed(100 + i)
        obs = _batch(b=4, h=32)
        act = torch.randint(0, 2, (4,), generator=g)
        olp = torch.full((4,), -0.69)
        adv = torch.randn(4, generator=g)
        ret = torch.ones(4)
        with torch.no_grad():
            ref_lp = masked_log_softmax(net.forward_value(obs)[0])
        loss, st = ppo_loss(net, ref_lp, obs, act, olp, adv, ret, 0.2, 0.5, 0.1, 0.01)
        assert torch.isfinite(loss)
        for k, v in st.items():
            assert isinstance(v, torch.Tensor) and v.dim() == 0 and not v.requires_grad
            old.setdefault(k, []).append(float(v.item()))  # old path: sync per minibatch
            sums[k] = v.detach().clone() if k not in sums else sums[k] + v.detach()
        n += 1
    new = {k: float((v / n).item()) for k, v in sums.items()}
    assert set(new) == {"pg", "vf", "ent", "kl_bc", "clipfrac", "approx_kl"}
    for k in new:
        assert abs(new[k] - float(np.mean(old[k]))) <= 1e-6, (k, new[k], np.mean(old[k]))


def test_compile_keeps_state_dict_keys():
    """torch.compile(model, dynamic=True) must not change saved state_dict keys
    (train_rl saves the original module's state_dict). Build only, no GPU."""
    try:
        import torch._dynamo as _dynamo  # noqa: F401
        compile = torch.compile
    except (ImportError, AttributeError):
        import pytest
        pytest.skip("torch.compile unavailable")
    emb = _emb()
    net = PolicyNet(emb, d=16, layers=1, heads=2, value=True, dropout=0.0)
    eager_keys = list(net.state_dict().keys())
    try:
        compiled = compile(net, dynamic=True)
        wrapped_keys = list(compiled._orig_mod.state_dict().keys())
    except Exception:
        import pytest
        pytest.skip("torch.compile unavailable on CPU")
    assert wrapped_keys == eager_keys
    assert not any(k.startswith("_orig_mod.") for k in eager_keys)


def test_slot_store_put_release():
    st = SlotStore(5, {"x": np.zeros((1, 3), np.int32)})
    s = st.put({"x": np.ones((2, 3), np.int32)}, np.array([1, 2]), np.zeros(2), np.zeros(2),
               np.full((2, 64), -0.5, np.float32))
    assert st.free == 3 and (st.obs["x"][s] == 1).all()
    assert st.ref_lp.shape == (5, 64) and st.ref_lp.dtype == np.float16
    assert np.allclose(st.ref_lp[s].astype(np.float32), -0.5, atol=1e-2)
    st.release(s)
    assert st.free == 5


def test_slot_store_concurrent_put_release():
    """Two threads x 2,000 put/release cycles: no duplicate handouts, no lost slots."""
    import threading

    cap = 256
    st = SlotStore(cap, {"x": np.zeros((1, 3), np.int32)})
    in_use: set[int] = set()
    mu = threading.Lock()
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for _ in range(n):
                s = st.put({"x": np.ones((1, 3), np.int32)}, np.array([1]), np.zeros(1),
                           np.zeros(1), np.full((1, 64), -0.5, np.float32))
                sid = int(s[0])
                with mu:
                    assert sid not in in_use, f"duplicate slot {sid}"
                    in_use.add(sid)
                with mu:
                    in_use.discard(sid)
                st.release(s)
        except BaseException as e:  # noqa: BLE001 - re-raised in the main thread
            with mu:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(2000,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0] if errors else None
    assert not in_use and st.free == cap
