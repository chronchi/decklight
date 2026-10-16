"""C++ replay fast path (speed plan E5): Env.replay / Env.replay_keys with the GIL released.

Compares against the per-step Python loop: identical obs (every array), done/winner,
choices/decisions/history, plus Determinizer.sample bit-identity old vs new, a GIL
parallelism check, and a micro benchmark (also printed by scripts/audit/replay_bench.py).
"""

import os
import random
import threading
import time

import numpy as np
import pytest

from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig
from ygo_zero.search.determinize import (
    Determinizer,
    SeenTracker,
    find_by_key_rank,
    obs_equal,
    prefix_keys_for,
    root_obs_equal,
)

PAIRS = [(0, 1), (2, 3), (0, 2)]


def _cfgs(decks):
    names = sorted(decks)
    return [DuelConfig(decks[names[a]], decks[names[b]], seed=100 + i) for i, (a, b) in enumerate(PAIRS)]


def _random_prefix(engine, cfg, seed, length):
    """Uniform-random prefix of `length` choices (or to duel end); returns (prefix, obss)."""
    rng = random.Random(seed)
    env = CppEnv(engine, cfg, True)
    obs = env.reset()
    obss = [obs]
    prefix = []
    while obs is not None and len(prefix) < length:
        legal = np.flatnonzero(obs["action_mask"]).tolist()
        c = rng.choice(legal)
        prefix.append(int(c))
        obs = env.step(c)
        obss.append(obs)
    return prefix, obss


def _state(env):
    return {
        "done": bool(env.done),
        "winner": None if env.winner is None else int(env.winner),
        "choices": list(env.choices),
        "decisions": int(env.decisions),
        "truncated": int(env.truncated),
        "hlen": int(env.history_len()),
        "hlog": np.asarray(env.history_log()),
    }


def test_replay_matches_step_loop(engine, decks):
    """replay(prefix) == reset + step per choice: obs every array, done/winner/choices/decisions/history."""
    for cfg in _cfgs(decks):
        prefix, _ = _random_prefix(engine, cfg, seed=5, length=400)
        assert len(prefix) >= 50, "prefix too short for a real comparison"
        # several prefix lengths incl. empty, mid-game and (possibly) terminal
        cuts = [0, 1, 7, len(prefix) // 2, len(prefix) - 1, len(prefix)]
        for n in cuts:
            ref = CppEnv(engine, cfg, True)
            obs_ref = ref.reset()
            for c in prefix[:n]:
                obs_ref = ref.step(c)
            got = CppEnv(engine, cfg, True)
            obs = got.replay_from_reset(prefix[:n])
            assert obs_equal(obs, obs_ref), f"obs differs at cut {n}"
            a, b = _state(ref), _state(got)
            assert a["done"] == b["done"] and a["winner"] == b["winner"], f"end differs at {n}"
            assert a["choices"] == b["choices"] and a["decisions"] == b["decisions"], f"counts differ at {n}"
            assert a["truncated"] == b["truncated"] and a["hlen"] == b["hlen"], f"hist differs at {n}"
            assert np.array_equal(a["hlog"], b["hlog"]), f"history_log differs at {n}"


def test_replay_out_of_range_raises_like_step(engine, decks):
    """An out-of-range index raises in replay exactly like in the step loop."""
    cfg = _cfgs(decks)[0]
    ref = CppEnv(engine, cfg, True)
    obs = ref.reset()
    n_legal = int(obs["action_mask"].sum())
    with pytest.raises(IndexError):
        ref.step(n_legal + 3)
    got = CppEnv(engine, cfg, True)
    with pytest.raises(IndexError):
        got.replay_from_reset([n_legal + 3])


def _python_keys_loop(engine, world, keys_ranks):
    """The old Python semantic loop; returns (obs, resolved, reason)."""
    env = CppEnv(engine, world, True)
    obs = env.reset()
    resolved = []
    for key, rank in keys_ranks:
        if obs is None:
            return obs, resolved, "replay_error"
        idx = find_by_key_rank(obs, key, rank)
        if idx is None:
            return obs, resolved, "no_key"
        resolved.append(idx)
        obs = env.step(idx)
    return obs, resolved, None


def test_replay_keys_matches_python_loop(engine, decks):
    """replay_keys == find_by_key_rank + step loop: resolved, reason, final obs.

    Uses determinized worlds (shuffled hidden cards) so some keys miss (no_key), and
    bumps ranks above the copy count to hit the take-last-copy path.
    """
    for gi, cfg in enumerate(_cfgs(decks)):
        prefix, _ = _random_prefix(engine, cfg, seed=21, length=150)
        assert len(prefix) >= 40
        pkeys = prefix_keys_for(engine, cfg, prefix, True)
        # live-game seen sets + root obs at the cut, like Determinizer
        env = CppEnv(engine, cfg, True)
        obs = env.reset()
        t = SeenTracker(0)
        t.update(env)
        for c in prefix:
            obs = env.step(c)
            t.update(env)
        me = int(env.player) if obs is not None else 0
        root_obs = obs
        assert root_obs is not None
        det = Determinizer(engine, cfg, True, me, prefix, set(t.seen), root_obs,
                           random.Random(f"rk:{gi}"), tries=4, prefix_keys=pkeys)
        checked = 0
        for trial in range(12):
            mains = det._propose_mains()
            if mains is None:
                mains = {0: list(det.true_main[0]), 1: list(det.true_main[1])}
            world = det._build_config(mains)
            # rank above the copy count on some entries (take-last-copy path)
            kr = [(k, r + (3 if (i % 5 == 0) else 0)) for i, (k, r) in enumerate(pkeys)]
            o_py, res_py, reason_py = _python_keys_loop(engine, world, kr)
            e2 = CppEnv(engine, world, True)
            o_cpp, res_cpp, reason_cpp = e2.replay_keys(kr)
            assert list(res_cpp) == res_py, f"resolved differ: {res_cpp} vs {res_py}"
            assert reason_cpp == reason_py, f"reason differs: {reason_cpp} vs {reason_py}"
            if reason_py is None:
                # all keys resolved: same final obs (None iff the duel ended exactly at the end)
                assert obs_equal(o_cpp, o_py), "final obs differs"
            else:
                # on failure the C++ path returns no obs (the caller discards it either way)
                assert o_cpp is None
            checked += 1
            if reason_py == "no_key" and trial >= 3:
                break
        assert checked >= 4


def test_determinizer_sample_bit_identical(engine, decks):
    """Determinizer.sample identical with the C++ fast path and the old Python loop."""
    import ygo_zero.search.determinize as D

    for gi, cfg in enumerate(_cfgs(decks)[:2]):
        prefix, _ = _random_prefix(engine, cfg, seed=33, length=120)
        assert len(prefix) >= 40
        pkeys = prefix_keys_for(engine, cfg, prefix, True)
        env = CppEnv(engine, cfg, True)
        obs = env.reset()
        trackers = {0: SeenTracker(0), 1: SeenTracker(1)}
        trackers[0].update(env)
        trackers[1].update(env)
        for c in prefix:
            obs = env.step(c)
            trackers[0].update(env)
            trackers[1].update(env)
        assert obs is not None
        me = int(env.player)
        outs = []
        for use_cpp in (False, True):
            D.USE_CPP_REPLAY = use_cpp
            try:
                det = Determinizer(engine, cfg, True, me, prefix, set(trackers[me].seen), obs,
                                   random.Random(f"ident:{gi}"), tries=8, prefix_keys=pkeys)
                got = [det.sample() for _ in range(3)]
                outs.append((
                    [(None if g is None else (g[0].seed, list(g[1]))) for g in got],
                    det.proposals, det.accepted, det.fallbacks, dict(det.reasons),
                ))
            finally:
                D.USE_CPP_REPLAY = True
        assert outs[0] == outs[1], f"sample differs old vs new: {outs[0]} vs {outs[1]}"
        # accepted worlds still pass the semantic root check under the new path
        D.USE_CPP_REPLAY = True
        det = Determinizer(engine, cfg, True, me, prefix, set(trackers[me].seen), obs,
                           random.Random(f"ident:{gi}"), tries=8, prefix_keys=pkeys)
        for _ in range(3):
            got = det.sample()
            if got is not None:
                world, resolved = got
                e = CppEnv(engine, world, True)
                o = e.replay_from_reset(resolved)
                assert root_obs_equal(o, obs)


def test_replay_releases_gil(engine, decks):
    """Two threads replaying a long prefix take < 1.6x one thread (GIL released)."""
    if (os.cpu_count() or 0) < 2:
        pytest.skip("needs >= 2 cores")
    cfg = _cfgs(decks)[0]
    prefix, _ = _random_prefix(engine, cfg, seed=7, length=400)
    assert len(prefix) >= 100
    prefix = prefix[:120]

    def work():
        e = CppEnv(engine, cfg, True)
        for _ in range(8):
            e.replay_from_reset(prefix)

    work()  # warm up (script cache, allocator)
    t0 = time.perf_counter()
    work()
    t1 = time.perf_counter()
    one = t1 - t0
    t0 = time.perf_counter()
    ts = [threading.Thread(target=work) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    two = time.perf_counter() - t0
    # 2 threads x the same work: with the GIL released both run in parallel on 2 cores.
    assert two < 1.6 * one, f"no GIL release: 1-thread {one:.2f}s, 2-thread {two:.2f}s"
