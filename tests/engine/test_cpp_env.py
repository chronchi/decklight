"""Contract: the C++ env produces exactly the Python reference env's observations, step for step."""

import random
import zlib

import numpy as np
import pytest

from decklight.engine.duel import DuelConfig
from decklight.engine.env import PyEnv

cpp_env = pytest.importorskip("decklight.engine.cpp_env", reason="extension not built: scripts/build_ext.sh")


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


def test_vecenv_max_decisions_truncates(engine, decks):
    deck_list = [decks[k] for k in sorted(decks)]
    vec = cpp_env.VecEnv(engine, deck_list, num_envs=4, num_threads=2, seed=3)
    vec.set_max_decisions(20)
    buf, rng, eps = vec.reset(), np.random.default_rng(0), []
    for _ in range(200):
        counts = buf["action_mask"].sum(1)
        vec.step((rng.random(4) * counts).astype(np.int32))
        done = buf["done"].copy()
        new = vec.pop_finished()
        assert len(new) == int(done.sum())
        eps += new
    assert eps and all(e.winner in (0, 1, 2, cpp_env.VecEnv.TRUNCATED) for e in eps)
    cut = [e for e in eps if e.winner == cpp_env.VecEnv.TRUNCATED]
    assert cut, "no game reached the 20-decision cap"
    for ep in cut[:3]:  # a cut game is a prefix of a real one: replays without ending
        env = PyEnv.replay(engine, vec.episode_config(ep), ep.choices)
        assert not env.done


def _policy_action(buf, i):
    count = int(buf["action_mask"][i].sum())
    assert count > 0
    return zlib.crc32(buf["actions"][i].tobytes() + buf["cards"][i].tobytes()) % count


def test_vecenv_async_matches_sync(engine, decks):
    deck_list = [decks[k] for k in sorted(decks)]
    n = len(deck_list)
    games = [(k % n, (2 * k + 1) % n, 1000 + k) for k in range(12)]

    def make():
        vec = cpp_env.VecEnv(engine, deck_list, num_envs=4, num_threads=3, seed=99)
        vec.set_schedule(games)
        vec.set_max_decisions(300)
        return vec

    sync_vec = make()
    buf = sync_vec.reset()
    sync_eps = []
    while sync_vec.remaining > 0:
        counts = buf["action_mask"].sum(1)
        actions = np.zeros(4, dtype=np.int32)
        for i in range(4):
            if int(buf["player"][i]) != -1:
                actions[i] = _policy_action(buf, i)
        sync_vec.step(actions)
        sync_eps += sync_vec.pop_finished()
    sync_map = {e.index: (e.winner, e.turns, tuple(e.choices)) for e in sync_eps}
    assert len(sync_map) == 12

    async_vec = make()
    buf = async_vec.async_reset()
    async_eps = []
    while True:
        ids = async_vec.recv(1, 4)
        if len(ids) == 0:
            break
        send_ids, send_acts = [], []
        for i in ids:
            if int(buf["player"][int(i)]) != -1:
                send_ids.append(int(i))
                send_acts.append(_policy_action(buf, int(i)))
        if send_ids:
            async_vec.send(send_ids, send_acts)
        async_eps += async_vec.pop_finished()
    async_eps += async_vec.pop_finished()
    async_map = {e.index: (e.winner, e.turns, tuple(e.choices)) for e in async_eps}
    assert len(async_map) == 12
    assert async_map == sync_map


def test_vecenv_async_misuse(engine, decks):
    deck_list = [decks[k] for k in sorted(decks)]
    vec = cpp_env.VecEnv(engine, deck_list, num_envs=2, num_threads=1, seed=5)
    assert len(vec.recv(1, 2)) == 0  # nothing outstanding: empty at once
    vec.async_reset()
    ids = vec.recv(2, 2)
    assert len(ids) == 2
    vec.send([int(ids[0])], [0])
    with pytest.raises(ValueError):  # id not HELD (already queued)
        vec.send([int(ids[0])], [0])
    with pytest.raises(RuntimeError):  # sync step while async work outstanding
        vec.step(np.zeros(2, dtype=np.int32))
    got = vec.recv(1, 2)
    assert len(got) == 1
    assert len(vec.recv(1, 2)) == 0
