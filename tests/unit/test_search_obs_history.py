"""Search with an observable-history product (Y, prereg 2026-11-11 value v2 step 0): the search envs (CppEnv: prefix
replay, clone + step, replay by option key) must give the same `obs_history` as the training env (PyEnv) on the same
game, so Y sees in search exactly what it saw in training."""

from pathlib import Path

import numpy as np
import pytest

from decklight.decks import load_ydk
from decklight.engine import cpp_env
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("obs_history", "obs_history_mask")


@pytest.fixture(scope="module")
def engine():
    return Engine()


def _same(a: dict, b: dict) -> bool:
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in KEYS)


def _cfg(seed: int) -> DuelConfig:
    decks = [load_ydk(p) for p in sorted((ROOT / "tests/fixtures/decks").glob("*.ydk"))]
    return DuelConfig(decks[seed % len(decks)], decks[(seed + 1) % len(decks)], seed=seed)


@pytest.mark.parametrize("seed", [3, 4])
def test_cpp_env_obs_history_matches_py_env(engine, seed):
    rng = np.random.default_rng(seed)
    py, cpp = PyEnv(engine, _cfg(seed)), CppEnv(engine, _cfg(seed))
    a, b, choices, nonempty = py.reset(), cpp.reset(), [], 0
    while a is not None and len(choices) < 200:
        assert b is not None and _same(a, b), f"obs_history differs at decision {len(choices)}"
        nonempty += int(np.asarray(a["obs_history_mask"]).sum() > 0)
        c = int(rng.integers(int(a["action_mask"].sum())))
        choices.append(c)
        a, b = py.step(c), cpp.step(c)
    assert b is None if a is None else True
    assert nonempty > 10, "the game never filled obs_history: the test checks nothing"


def test_clone_and_replay_keep_obs_history(engine):
    rng = np.random.default_rng(5)
    env = CppEnv(engine, _cfg(5))
    obs, prefix, checked = env.reset(), [], 0
    while obs is not None and len(prefix) < 120:
        c = int(rng.integers(int(obs["action_mask"].sum())))
        root = env.clone() if cpp_env.SEARCH_CLONE and len(prefix) % 10 == 9 else None
        prefix.append(c)
        obs = env.step(c)
        if len(prefix) % 10 == 0 and obs is not None:  # WindowSearch's branch: replay the prefix (clone on the Rust core)
            rp = CppEnv(engine, _cfg(5)).replay_from_reset(prefix)
            assert rp is not None and _same(obs, rp), f"replay obs_history differs after {len(prefix)} choices"
            if root is not None:
                assert _same(obs, root.step(c)), f"clone obs_history differs after {len(prefix)} choices"
            checked += 1
    assert checked >= 5
