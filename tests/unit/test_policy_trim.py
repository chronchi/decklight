"""PolicyNet.trim: dropping batch padding (trim_obs) must not change logits or values."""

from pathlib import Path

import numpy as np
import pytest
import torch

from decklight.decks import load_ydk
from decklight.engine.cpp_env import VecEnv
from decklight.engine.duel import Engine
from decklight.model.policy import PolicyNet, trim_obs
from decklight.rl.ppo import gather, to_torch

ROOT = Path(__file__).resolve().parents[2]


def _emb() -> dict:
    rng = np.random.default_rng(0)
    codes = np.array(
        sorted({c for p in (ROOT / "tests/fixtures/decks").glob("*.ydk") for c in load_ydk(p).main})
    )
    n = len(codes)
    return {"codes": codes, "card": rng.standard_normal((n, 8)).astype(np.float32),
            "effects": rng.standard_normal((n, 2, 8)).astype(np.float32), "n_effects": np.full(n, 2)}  # fmt: skip


def _batches(steps: int = 40, envs: int = 32):
    decks = [load_ydk(p) for p in sorted((ROOT / "tests/fixtures/decks").glob("*.ydk"))]
    vec = VecEnv(Engine(), decks, envs, 2, seed=5)
    buf = vec.reset()
    rng = np.random.default_rng(1)
    for t in range(steps):
        if t % 10 == 9:
            yield {k: np.array(v) for k, v in buf.items()}
        vec.step((rng.random(envs) * buf["action_mask"].sum(1)).astype(np.int32))


@pytest.mark.parametrize("hist,known", [(0, False), (64, True)])
def test_trim_matches_full_inputs(hist, known):
    torch.manual_seed(0)
    net = PolicyNet(_emb(), d=16, layers=2, heads=2, dropout=0.0, hist=hist, known=known, value=True).eval()
    n_batches = 0
    for buf in _batches():
        obs = to_torch(gather(buf, np.arange(len(buf["cards"])), hist, known), torch.device("cpu"))
        with torch.no_grad():
            net.trim = False
            l_full, v_full = net.forward_value(obs)
            net.trim = True
            l_trim, v_trim = net.forward_value(obs)
        assert l_trim.shape == l_full.shape
        assert torch.equal(torch.isfinite(l_trim), torch.isfinite(l_full))
        fin = torch.isfinite(l_full)
        assert torch.allclose(l_trim[fin], l_full[fin], atol=1e-5, rtol=1e-4)
        assert torch.allclose(v_trim, v_full, atol=1e-5, rtol=1e-4)
        t, _ = trim_obs(obs)
        assert t["cards"].shape[1] < obs["cards"].shape[1] and t["actions"].shape[1] < obs["actions"].shape[1]
        n_batches += 1
    assert n_batches == 4
