"""Contract: the C++ env produces exactly the Python reference env's observations, step for step."""

import random

import numpy as np
import pytest

from ygo_zero.engine.duel import DuelConfig
from ygo_zero.engine.env import PyEnv

cpp_env = pytest.importorskip("ygo_zero.engine.cpp_env", reason="extension not built: scripts/build_ext.sh")


def lockstep(engine, config, seed, max_steps=100_000):
    """Play both envs with the same random choices; return the first mismatch (or None) and the step count."""
    rng = random.Random(seed)
    py, cc = PyEnv(engine, config), cpp_env.CppEnv(engine, config)
    a, b = py.reset(), cc.reset()
    for step in range(max_steps):
        if (a is None) != (b is None):
            return f"step {step}: one env ended", step
        if a is None:
            break
        for k in a:
            if not np.array_equal(a[k], b[k]):
                diff = np.argwhere(a[k] != b[k])[:5].tolist()
                return f"step {step}: {k} differs at {diff}", step
        i = rng.randrange(int(a["action_mask"].sum()))
        a, b = py.step(i), cc.step(i)
    if (py.winner, py.decisions) != (cc.winner, cc.decisions):
        return f"end differs: {(py.winner, py.decisions)} vs {(cc.winner, cc.decisions)}", step
    return None, step


@pytest.mark.parametrize("seed", range(6))
def test_cpp_matches_python(engine, decks, seed):
    names = sorted(decks)
    config = DuelConfig(decks[names[seed % 4]], decks[names[(seed + 2) % 4]], seed=100 + seed)
    mismatch, _ = lockstep(engine, config, seed)
    assert mismatch is None, mismatch
