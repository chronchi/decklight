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


def _run_vec(engine, decks, threads, steps, seed=7, n=6):
    vec = cpp_env.VecEnv(engine, decks, num_envs=n, num_threads=threads, seed=seed)
    buf, rng = vec.reset(), np.random.default_rng(0)
    first = {k: v.copy() for k, v in buf.items()}
    eps = []
    for _ in range(steps):
        counts = buf["action_mask"].sum(1)
        vec.step((rng.random(n) * counts).astype(np.int32))
        eps += vec.pop_finished()
    return vec, first, eps


def test_vecenv_episodes_replay_and_are_thread_independent(engine, decks):
    deck_list = [decks[k] for k in sorted(decks)]
    vec, first, eps = _run_vec(engine, deck_list, threads=3, steps=600)
    assert eps, "no episode finished"
    for ep in eps[:4]:  # every finished game replays to the same result in the reference env
        env = PyEnv.replay(engine, vec.episode_config(ep), ep.choices)
        assert env.done and env.winner == ep.winner and env.duel.turn == ep.turns
    # same seed, different thread count: same games
    _, first1, eps1 = _run_vec(engine, deck_list, threads=1, steps=600)
    assert all(np.array_equal(first[k], first1[k]) for k in first)
    key = [(e.deck0, e.deck1, e.seed, e.winner, tuple(e.choices)) for e in eps]
    assert sorted(key) == sorted((e.deck0, e.deck1, e.seed, e.winner, tuple(e.choices)) for e in eps1)
