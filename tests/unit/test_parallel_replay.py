"""Thread-parallel world checks and branch replays (speed plan E6, exact).

With `replay_threads > 1` every game must be bit-identical to `replay_threads = 1`:
same worlds, same resolved indices, same counters, same choices, same logs.
"""

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig
from decklight.search.determinize import (
    Determinizer,
    SeenTracker,
    key_rank_of_choice,
    obs_equal,
)


def _cfg(decks, seed):
    names = sorted(decks)
    return DuelConfig(decks[names[0]], decks[names[1]], seed=seed)


def _decision_points(engine, cfg, seed, min_turn=2):
    """(prefix, prefix_keys, me, root_obs, seen) at turn>=min_turn decisions of a random game."""
    rng = random.Random(seed)
    env = CppEnv(engine, cfg, True)
    obs = env.reset()
    t0, t1 = SeenTracker(0), SeenTracker(1)
    t0.update(env)
    t1.update(env)
    keys: list = []
    pts = []
    while obs is not None and len(env.choices) < 600:
        if int(obs["global"][2]) >= min_turn:
            pts.append((list(env.choices), list(keys), int(env.player), obs,
                        {0: set(t0.seen), 1: set(t1.seen)}))
        legal = np.flatnonzero(obs["action_mask"]).tolist()
        c = rng.choice(legal)
        keys.append(key_rank_of_choice(obs, int(c)))
        obs = env.step(c)
        t0.update(env)
        t1.update(env)
    return pts


def _det(engine, cfg, pt, seed_str, tries):
    prefix, pkeys, me, root_obs, seen = pt
    return Determinizer(engine, cfg, True, me, prefix, seen[me], root_obs,
                        random.Random(seed_str), tries=tries, prefix_keys=pkeys)


def _snapshot(det, results):
    """Comparable snapshot: per result (mains or None, resolved) + counters + rng state."""
    snap = []
    for got in results:
        if got is None:
            snap.append(None)
        else:
            world, resolved = got
            snap.append(((list(world.main_of(0)), list(world.main_of(1))), [int(x) for x in resolved]))
    return (snap, det.proposals, det.accepted, det.fallbacks, dict(det.reasons), det.rng.getstate())


@pytest.mark.parametrize("workers", [2, 4])
@pytest.mark.parametrize("tries,samples", [(8, 3), (1, 4)])
def test_sample_many_matches_sequential(engine, decks, workers, tries, samples):
    """sample_many(n, pool) == [sample() for _ in range(n)]: worlds, resolved indices,
    counters, reasons and the rng state afterwards. tries=8 sees rejections, tries=1
    sees fallbacks."""
    cfg = _cfg(decks, 21)
    pts = _decision_points(engine, cfg, seed=100)
    assert len(pts) >= 6
    with ThreadPoolExecutor(max_workers=workers) as pool:
        seen_reject = seen_fallback = False
        for pi, pt in enumerate(pts[:6]):
            seq = _det(engine, cfg, pt, f"e6:{pi}", tries)
            want = _snapshot(seq, [seq.sample() for _ in range(samples)])
            par = _det(engine, cfg, pt, f"e6:{pi}", tries)
            got = _snapshot(par, par.sample_many(samples, pool))
            assert got == want, f"point {pi} differs with {workers} workers"
            seen_reject |= (seq.reasons["obs_mismatch"] + seq.reasons["no_key"]
                            + seq.reasons["replay_error"]) > 0
            seen_fallback |= seq.fallbacks > 0
    assert seen_reject, "no rejection observed: the parallel scan path is untested"
    assert tries == 8 or seen_fallback, "tries=1 never fell back"


def test_sample_many_none_pool_is_sequential(engine, decks):
    """pool=None runs today's code path: identical to repeated sample()."""
    cfg = _cfg(decks, 22)
    pts = _decision_points(engine, cfg, seed=101)
    for pi, pt in enumerate(pts[:3]):
        seq = _det(engine, cfg, pt, f"e6n:{pi}", 4)
        want = _snapshot(seq, [seq.sample() for _ in range(3)])
        par = _det(engine, cfg, pt, f"e6n:{pi}", 4)
        assert _snapshot(par, par.sample_many(3, None)) == want


def _tiny_forward(decks):
    from decklight.model.bc_data import OBS_KEYS
    from decklight.model.policy import PolicyNet

    names = sorted(decks)
    codes = sorted({c for n in names for c in list(decks[n].main) + list(decks[n].extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    net = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).eval()

    @torch.no_grad()
    def forward(obs_list):
        t = {k: torch.from_numpy(np.ascontiguousarray(np.stack([o[k] for o in obs_list])))
             for k in OBS_KEYS}
        return net(t).numpy()

    return forward


def _play(engine, decks, forward, pool, game_seed):
    from decklight.search.play import argmax_select, search_game_gen
    from decklight.search.turn1 import drive_many
    from decklight.search.window import WindowSearch

    def never(buf):
        raise AssertionError("unused")

    names = sorted(decks)
    cfg = DuelConfig(decks[names[2]], decks[names[3]], seed=8)
    ws = WindowSearch(engine, never, k=2, horizon=1, samples=2, seed=0, pool=pool)
    gen = search_game_gen(engine, cfg, ws, 0, argmax_select, game_seed=game_seed,
                          own_turn1=True, conf_skip=1.01, own_frac=1.0,
                          honest=True, det_tries=2, det_fallback="true", max_searches=4)
    return dict(drive_many([(0, gen)], forward, 2))[0]


def test_game_pool3_matches_pool1(engine, decks):
    """replay_threads 1 vs 3 on 3 honest deals (max_searches=4, the test_search_prof
    pattern): identical choices, log picks/scores and det counters."""
    fw = _tiny_forward(decks)
    with ThreadPoolExecutor(max_workers=3) as pool3:
        for seed in (0, 1, 2):
            off = _play(engine, decks, fw, None, seed)
            on = _play(engine, decks, fw, pool3, seed)
            assert off.n_searched >= 1 and off.det_proposals > 0
            assert (on.choices, on.actors) == (off.choices, off.actors)
            assert (on.n_searched, on.n_override, on.det_proposals, on.det_accepted,
                    on.det_fallbacks, on.det_skipped) == \
                   (off.n_searched, off.n_override, off.det_proposals, off.det_accepted,
                    off.det_fallbacks, off.det_skipped)
            assert [(e["cands"], e["chosen"], e["scores"], e["samples"]) for e in on.log] == \
                   [(e["cands"], e["chosen"], e["scores"], e["samples"]) for e in off.log]


def test_branch_setup_pool_matches_sequential(engine, decks):
    """_branch_envs with a pool returns the same obs list (every array) as the loop."""
    from decklight.search.window import WindowSearch

    cfg = _cfg(decks, 3)
    rng = random.Random(9)
    env = CppEnv(engine, cfg, True)
    obs = env.reset()
    prefix = []
    while obs is not None and len(prefix) < 40:
        legal = np.flatnonzero(obs["action_mask"]).tolist()
        c = rng.choice(legal)
        prefix.append(int(c))
        obs = env.step(c)
    assert obs is not None
    probe = CppEnv(engine, cfg, True)
    o = probe.reset()
    for c in prefix:
        o = probe.step(c)
    legal = np.flatnonzero(o["action_mask"]).tolist()[:2]
    assert len(legal) == 2
    worlds = [(cfg, prefix), (cfg, prefix)]
    branches = [(prefix + [a], j, a, s) for j, a in enumerate(legal) for s in range(3)]

    def never(buf):
        raise AssertionError("unused")

    ws_seq = WindowSearch(engine, never, k=2, horizon=1, samples=3, seed=0)
    _, obs_seq = ws_seq._branch_envs(cfg, True, branches, worlds)
    with ThreadPoolExecutor(max_workers=3) as pool:
        ws_par = WindowSearch(engine, never, k=2, horizon=1, samples=3, seed=0, pool=pool)
        _, obs_par = ws_par._branch_envs(cfg, True, branches, worlds)
    assert len(obs_par) == len(obs_seq) == len(branches)
    for i, (a, b) in enumerate(zip(obs_par, obs_seq, strict=True)):
        assert obs_equal(a, b), f"branch {i} obs differs"
