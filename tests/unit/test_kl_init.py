"""KL anchor to the init policy in train_bc (--kl-init)."""

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_kl_zero_at_init_and_finite_with_mask():
    train_bc = _load("train_bc")
    torch.manual_seed(0)
    logits = torch.randn(4, 8)
    logits[:, 5:] = float("-inf")  # 5 legal options
    mask = torch.zeros(4, 8, dtype=torch.bool)
    mask[:, :5] = True
    kl = train_bc.kl_to_init(logits, logits.clone(), mask)
    torch.testing.assert_close(kl, torch.zeros(4), atol=1e-6, rtol=0)
    # shifted logits: finite, non-negative
    kl2 = train_bc.kl_to_init(logits + torch.randn(4, 8) * 0.1, logits, mask)
    assert torch.isfinite(kl2).all() and (kl2 >= 0).all()
    # illegal-only columns at -inf in both must not leak NaN
    assert not torch.isnan(kl2).any()


def test_kl_matches_reference_math():
    train_bc = _load("train_bc")
    logits = torch.tensor([[1.0, 2.0, 0.5, float("-inf")]])
    ref = torch.tensor([[0.5, 0.5, 1.5, float("-inf")]])
    mask = torch.tensor([[True, True, True, False]])
    got = train_bc.kl_to_init(logits, ref, mask).item()
    import math

    def sm(x):
        e = [math.exp(v) for v in x]
        s = sum(e)
        return [v / s for v in e]

    p, q = sm([0.5, 0.5, 1.5]), sm([1.0, 2.0, 0.5])
    want = sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q))
    assert got == pytest.approx(want, abs=1e-6)


def _tiny_net(seed=0):
    from ygo_zero.model.policy import PolicyNet

    rng = np.random.default_rng(42)
    codes = np.arange(1, 50)
    n = len(codes)
    emb = {"codes": codes, "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}
    torch.manual_seed(seed)
    return PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).train()


def _tiny_obs(model, batch=8, seed=1):
    """Synthetic obs straight through PolicyNet: random legal masks, random rows."""
    from ygo_zero.model.bc_data import OBS_KEYS

    torch.manual_seed(seed)
    shapes = {"cards": (batch, 160, 21), "card_mask": (batch, 160), "global": (batch, 24),
              "chain": (batch, 16, 5), "chain_mask": (batch, 16), "actions": (batch, 64, 10),
              "action_mask": (batch, 64), "history": (batch, 32, 13), "history_mask": (batch, 32)}
    obs = {}
    for k in OBS_KEYS:
        s = shapes[k]
        if k.endswith("_mask"):
            m = torch.zeros(s, dtype=torch.bool)
            m[:, :4] = True  # 4 legal entries each
            obs[k] = m
        elif k in ("cards", "chain", "actions", "history"):
            obs[k] = torch.randint(0, 4, s)
        else:
            obs[k] = torch.randint(0, 4, s)
    # keep action rows in embedding range
    obs["actions"] = torch.randint(0, 4, (batch, 64, 10))
    return obs


def _drift(obs_fn, coef, steps=15, seed=0):
    train_bc = _load("train_bc")
    net = _tiny_net(seed)
    ref = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    # fixed target: always option 0, while the init is random -> training drifts
    for _ in range(steps):
        obs = obs_fn()
        logits = net(obs)
        per = train_bc.set_loss(logits.float(), _onehot(logits.shape, 0))
        loss = per.mean()
        if coef:
            with torch.no_grad():
                rl = ref(obs)
            kl = train_bc.kl_to_init(logits.float(), rl.float(), obs["action_mask"])
            loss = loss + coef * kl.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        kl_fin = train_bc.kl_to_init(net(obs_fn()).float(), ref(obs_fn()).float(),
                                     obs_fn()["action_mask"]).mean().item()
    return kl_fin


def _onehot(shape, col):
    eq = torch.zeros(shape, dtype=torch.bool)
    eq[:, col] = True
    return eq


def test_large_coef_holds_model_closer_to_init():
    fixed = _tiny_obs(None)
    obs_fn = lambda: {k: v.clone() for k, v in fixed.items()}  # noqa: E731
    kl_free = _drift(obs_fn, coef=0.0)
    kl_anch = _drift(obs_fn, coef=100.0)
    assert np.isfinite(kl_free) and np.isfinite(kl_anch)
    assert kl_anch < kl_free, f"anchored KL {kl_anch:.4f} should be < free KL {kl_free:.4f}"


def test_kl_init_requires_init_flag(monkeypatch):
    import subprocess

    r = subprocess.run(
        ["uv", "run", "python", "scripts/train_bc.py", "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert "kl-init" in r.stdout
    # --kl-init without --init exits 2 (arg validation runs before any I/O)
    train_bc = _load("train_bc")
    monkeypatch.setattr(sys, "argv",
                        ["train_bc.py", "--shards", "x", "--emb", "y", "--out", "z", "--kl-init", "0.5"])
    with pytest.raises(SystemExit) as e:
        train_bc.main()
    assert e.value.code == 2
