"""Honest (determinized) search worlds (src/ygo_zero/search/determinize.py).

Engine only (uniform random over legal options, the search tests' "random-policy games"); a tiny
d=16 net only where a policy is needed (test 5, as the existing search tests).
"""

import random

import numpy as np

from ygo_zero.engine import constants as C
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig
from ygo_zero.search.determinize import (
    Determinizer, SeenTracker, _Deck, find_by_key_rank, key_rank_of_choice,
    obs_equal, prefix_keys_for, root_obs_equal,
)

LOCATIONS = (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE,
             C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA)


def _cfg(decks, seed):
    names = sorted(decks)
    return DuelConfig(decks[names[0]], decks[names[1]], seed=seed)


def _random_game(engine, cfg, seed, max_steps=400):
    """Uniform-random legal play; returns (env, obs_list) with the obs after each prefix."""
    rng = random.Random(seed)
    env = CppEnv(engine, cfg, True)
    obs = env.reset()
    obss = [obs]
    while obs is not None and len(env.choices) < max_steps:
        legal = np.flatnonzero(obs["action_mask"]).tolist()
        obs = env.step(rng.choice(legal))
        obss.append(obs)
    return env, obss


def _identity_config(cfg):
    """shuffle=False + main_of orders: the same add order the core saw (same RNG)."""
    return DuelConfig(
        _Deck(cfg.main_of(0), list(cfg.deck0.extra)),
        _Deck(cfg.main_of(1), list(cfg.deck1.extra)),
        seed=cfg.seed, flags=cfg.flags, lp=cfg.lp, hand=cfg.hand, draw=cfg.draw,
        setup=cfg.setup, shuffle=False,
    )


def test_slot_map_codes_match_main_of(engine, decks):
    """After reset, every main-deck card's code equals main_of(team)[slot]."""
    cfg = _cfg(decks, 1)
    env = CppEnv(engine, cfg, True)
    env.reset()
    cards = env.cards()
    assert (cards["cardid"] != 0).all() and len(set(cards["cardid"].tolist())) == len(cards["code"])
    for team in (0, 1):
        main = cfg.main_of(team)
        m = (cards["player"] == team) & (
            (cards["location"] == C.LOCATION_DECK) | (cards["location"] == C.LOCATION_HAND)
        )
        ids = sorted(cards["cardid"][m].tolist())
        assert len(ids) == len(main)
        assert ids == list(range(ids[0], ids[0] + len(main)))  # consecutive creation ids
        by_id = dict(zip(cards["cardid"][m].tolist(), cards["code"][m].tolist()))
        assert [by_id[i] for i in ids] == main


def test_identity_world_replays_bit_identically(engine, decks):
    """shuffle=False + main_of orders replays a 200-step random game, identical obs every step.

    The core RNG is seeded from config.seed only (seed_words); shuffle just changes the add
    order, which main_of already fixed — so the same seed replays bit-identically.
    """
    cfg = _cfg(decks, 11)
    env, obss = _random_game(engine, cfg, seed=5, max_steps=200)
    assert len(env.choices) >= 50  # a real prefix, not a turn-1 fizzle
    ide = _identity_config(cfg)
    env2 = CppEnv(engine, ide, True)
    obs = env2.reset()
    assert obs_equal(obs, obss[0])
    for k, c in enumerate(env.choices):
        obs = env2.step(c)
        assert obs_equal(obs, obss[k + 1]), f"obs differs at step {k}"
    assert (obs is None) == (obss[-1] is None)
    if obs is None:
        assert int(env2.winner) == int(env.winner)


def _decision_points(engine, cfg, seed, n=25):
    """(prefix, prefix_keys, me, root_obs, trackers) at turn>=2 decision points of a random game."""
    rng = random.Random(seed)
    env = CppEnv(engine, cfg, True)
    obs = env.reset()
    t0, t1 = SeenTracker(0), SeenTracker(1)
    t0.update(env)
    t1.update(env)
    keys: list = []
    pts = []
    while obs is not None and len(env.choices) < 600:
        if int(obs["global"][2]) >= 2:
            pts.append((list(env.choices), list(keys), int(env.player), obs, {0: set(t0.seen), 1: set(t1.seen)}))
        legal = np.flatnonzero(obs["action_mask"]).tolist()
        c = rng.choice(legal)
        keys.append(key_rank_of_choice(obs, int(c)))
        obs = env.step(c)
        t0.update(env)
        t1.update(env)
    return pts


def test_identity_world_semantic_replay_reproduces_prefix(engine, decks):
    """Semantic replay of the true config reproduces the true prefix indices exactly."""
    cfg = _cfg(decks, 31)
    rng = random.Random(7)
    env = CppEnv(engine, cfg, True)
    obs = env.reset()
    prefix, pkeys = [], []
    while obs is not None and len(prefix) < 120:
        legal = np.flatnonzero(obs["action_mask"]).tolist()
        c = rng.choice(legal)
        pkeys.append(key_rank_of_choice(obs, int(c)))
        prefix.append(int(c))
        obs = env.step(c)
    assert len(prefix) >= 50
    assert prefix_keys_for(engine, cfg, prefix, True) == pkeys
    probe = CppEnv(engine, cfg, True)
    o = probe.reset()
    resolved = []
    for key, rank in pkeys:
        idx = find_by_key_rank(o, key, rank)
        assert idx is not None
        resolved.append(idx)
        o = probe.step(idx)
    assert resolved == prefix


def _table(engine, cfg, prefix):
    env = CppEnv(engine, cfg, True)
    env.reset()
    for c in prefix:
        env.step(c)
    cards = env.cards()
    return {(int(cards["player"][i]), int(cards["cardid"][i])): int(cards["code"][i])
            for i in range(len(cards["code"]))}


def test_determinizer_accepts_consistent_worlds(engine, decks):
    """Every accepted world matches `me`'s root obs (meaning columns); seen cards keep their codes;
    over >= 20 decision points at least one accepted world changes the hidden hand."""
    cfgs = [_cfg(decks, s) for s in (21, 22)]
    n_pts, n_diff, n_acc = 0, 0, 0
    for gi, cfg in enumerate(cfgs):
        for prefix, pkeys, me, root_obs, seen in _decision_points(engine, cfg, seed=100 + gi):
            if n_pts >= 24:
                break
            det = Determinizer(engine, cfg, True, me, prefix, seen[me], root_obs,
                               random.Random(f"test:{gi}:{n_pts}"), tries=8,
                               prefix_keys=pkeys)
            for _ in range(2):
                got = det.sample()
                if got is None:
                    continue
                world, resolved = got
                n_acc += 1
                # root obs matches under the semantic comparison
                env = CppEnv(engine, world, True)
                obs = env.reset()
                for c in resolved:
                    obs = env.step(c)
                assert root_obs_equal(obs, root_obs)
                # cards in `seen` keep their codes (cardids are creation-ordered, same in both)
                true_tab, world_tab = _table(engine, cfg, prefix), _table(engine, world, resolved)
                for cid in seen[me]:
                    keys = [k for k in true_tab if k[1] == cid]
                    assert keys, f"seen cardid {cid} missing from true table"
                    for k in keys:
                        assert world_tab[k] == true_tab[k], f"seen card {k} changed code"
                # opponent hidden hand
                opp = 1 - me
                env_t, env_w = CppEnv(engine, cfg, True), CppEnv(engine, world, True)
                env_t.reset()
                env_w.reset()
                for c in prefix:
                    env_t.step(c)
                for c in resolved:
                    env_w.step(c)
                ct, cw = env_t.cards(), env_w.cards()
                ht = sorted(int(ct["code"][i]) for i in range(len(ct["code"]))
                            if int(ct["player"][i]) == opp and int(ct["location"][i]) == C.LOCATION_HAND
                            and not bool(ct["known"][i]))
                hw = sorted(int(cw["code"][i]) for i in range(len(cw["code"]))
                            if int(cw["player"][i]) == opp and int(cw["location"][i]) == C.LOCATION_HAND
                            and not bool(cw["known"][i]))
                if ht != hw:
                    n_diff += 1
            n_pts += 1
        if n_pts >= 24:
            break
    assert n_pts >= 20, f"only {n_pts} decision points"
    assert n_acc > 0, "no world accepted"
    assert n_diff >= 1, "no accepted world changed the opponent hidden hand"


def test_semantic_accepts_more_than_index(engine, decks):
    """Over >= 20 decision points, semantic replay accepts more proposals than index replay."""
    cfgs = [_cfg(decks, s) for s in (21, 22)]
    n_pts = 0
    sem_p = sem_a = 0
    idx_p = idx_a = 0
    for gi, cfg in enumerate(cfgs):
        for prefix, pkeys, me, root_obs, seen in _decision_points(engine, cfg, seed=300 + gi):
            if n_pts >= 24:
                break
            d_sem = Determinizer(engine, cfg, True, me, prefix, seen[me], root_obs,
                                 random.Random(f"sem:{gi}:{n_pts}"), tries=8,
                                 prefix_keys=pkeys)
            for _ in range(2):
                if d_sem.sample() is not None:
                    pass
            d_idx = Determinizer(engine, cfg, True, me, prefix, seen[me], root_obs,
                                 random.Random(f"sem:{gi}:{n_pts}"), tries=8)
            for _ in range(2):
                if d_idx.sample() is not None:
                    pass
            sem_p += d_sem.proposals
            sem_a += d_sem.accepted
            idx_p += d_idx.proposals
            idx_a += d_idx.accepted
            n_pts += 1
        if n_pts >= 24:
            break
    assert n_pts >= 20, f"only {n_pts} decision points"
    assert sem_p > 0 and idx_p > 0
    assert sem_a / sem_p > idx_a / idx_p, (
        f"semantic {sem_a}/{sem_p}={sem_a / sem_p:.3f} not above index {idx_a}/{idx_p}={idx_a / idx_p:.3f}"
    )


def test_rollouts_worlds_none_unchanged_and_true_world_matches(engine, decks):
    """rollouts_gen(worlds=None) is today's behaviour; worlds=[(true config, true prefix)] scores the same."""
    from ygo_zero.search.turn1 import drive_many
    from ygo_zero.search.window import WindowSearch

    cfg = _cfg(decks, 3)
    env, _ = _random_game(engine, cfg, seed=9, max_steps=40)
    prefix = list(env.choices[:20])
    probe = CppEnv(engine, cfg, True)
    o = probe.reset()
    for c in prefix:
        o = probe.step(c)
    assert o is not None
    me = int(probe.player)
    turn = 3

    def never(buf):
        raise AssertionError("unused")

    def run(worlds, samples):
        ws = WindowSearch(engine, never, k=2, horizon=1, samples=samples, seed=0)
        # two real candidates: first two legal options at the root
        e0 = CppEnv(engine, cfg, True)
        o = e0.reset()
        for c in prefix:
            o = e0.step(c)
        legal = np.flatnonzero(o["action_mask"]).tolist()[:2]
        branches = [(prefix + [a], j, a, s) for j, a in enumerate(legal) for s in range(samples)]
        gen = ws.rollouts_gen(cfg, True, branches, me, turn_start=turn, game_seed=0,
                              decision_k=0, worlds=worlds)
        fwd = lambda obs_list: np.zeros((len(obs_list), 64))  # noqa: E731
        return dict(drive_many([(0, gen)], fwd, 2))[0]

    base = run(None, 1)
    assert isinstance(base, dict)
    same = run([(cfg, prefix)], 1)
    assert same.keys() == base.keys()
    for k in base:
        assert same[k] == base[k], f"{k}: {same[k]} != {base[k]}"
    # multi-sample averages match too
    assert run([(cfg, prefix), (cfg, prefix)], 2) == run(None, 2)


def test_search_game_honest_ends_with_det_counters(engine, decks):
    """search_game_gen with honest on, under drive_many with the tiny net: one game ends,
    det counters > 0."""
    import torch

    from ygo_zero.model.bc_data import OBS_KEYS
    from ygo_zero.model.policy import PolicyNet
    from ygo_zero.search.play import argmax_select, search_game_gen
    from ygo_zero.search.turn1 import drive_many
    from ygo_zero.search.window import WindowSearch

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

    def never(buf):
        raise AssertionError("unused")

    cfg = DuelConfig(decks[names[2]], decks[names[3]], seed=8)
    ws = WindowSearch(engine, never, k=2, horizon=1, samples=1, seed=0)
    gen = search_game_gen(engine, cfg, ws, 0, argmax_select, game_seed=0,
                          own_turn1=True, conf_skip=1.01, own_frac=1.0, honest=True)
    res = dict(drive_many([(0, gen)], forward, 2))[0]
    assert not res.truncated and res.winner in (0, 1)
    assert res.n_searched >= 1
    assert res.det_proposals > 0 and (res.det_accepted + res.det_fallbacks) > 0
