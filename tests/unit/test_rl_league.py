"""League rollouts: PFSP mix, snapshots, stub ygo-agent, LSTM state hygiene (CPU, tiny nets)."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from decklight.decks import load_ydk
from decklight.engine.cpp_env import VecEnv
from decklight.model.policy import PolicyNet
from decklight.rl.league import BASE, SELF, YGO, League
from decklight.rl.ppo import SlotStore, gather
from decklight.rl.rollout import Collector

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"

_TRAIN_RL = None


def _train_rl_mod():
    global _TRAIN_RL
    if _TRAIN_RL is None:
        p = Path(__file__).resolve().parents[2] / "scripts" / "train_rl.py"
        spec = importlib.util.spec_from_file_location("train_rl_league_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _TRAIN_RL = mod
    return _TRAIN_RL


def _tiny(decks, value: bool):
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    return PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, value=value).eval()


class StubYgo:
    """First-legal-option ygo-agent stand-in: per-row episode tokens, reset tracking."""

    def __init__(self):
        self.calls = 0
        self.seen_keys: list[int] = []
        self.resets: list[int] = []

    def __call__(self, buf, rows, keys=None):
        assert keys is not None and len(keys) == len(rows), "league must pass episode tokens"
        self.calls += 1
        self.seen_keys.extend(int(k) for k in keys)
        return np.zeros(len(np.asarray(rows)), np.int32)

    def reset_keys(self, game_ids):
        self.resets.extend(int(g) for g in game_ids)


def _league_collector(engine, decks, seed=0, p_self=0.25, p_ygo=0.25, snaps=2, max_dec=60):
    dev = torch.device("cpu")
    model, ref = _tiny(decks, True), _tiny(decks, False)
    league = League(p_self, p_ygo, 6, np.random.default_rng(seed + 999))
    for k in range(snaps):
        league.add_snapshot(f"snap:{k}", _tiny(decks, False))
    ygo = StubYgo()
    vec = VecEnv(engine, decks, 8, 2, seed=seed)
    vec.set_max_decisions(max_dec)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False))
    col = Collector(vec, model, ref, store, 0, False, dev, 0.3, -0.5, 2,
                    np.random.default_rng(seed), "fp32", league, ygo)
    return col, store, league, ygo


def test_league_mix_covers_all_types(engine):
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    col, _, _, ygo = _league_collector(engine, decks)
    want = {SELF, BASE, YGO, "snap:0", "snap:1"}
    seen: set[str] = set()
    st = {}
    for _ in range(6):
        _, st = col.collect(400)
        seen |= {n for n, d in st["league"].items() if d["games"] > 0}
        if want <= seen:
            break
    assert want <= seen, seen
    assert ygo.calls > 0
    assert st["opp_games"] > 0 and st["opp_learner_samples"] > 0
    assert st["bc_games"] == st["league"][BASE]["games"]  # bc_* still mean vs base
    assert st["t_ga"] >= 0.0


def test_opponent_rows_never_stored(engine):
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    col, _, _, _ = _league_collector(engine, decks, seed=1, p_self=0.0, p_ygo=0.0)
    finished, st = col.collect(500)
    total = sum(len(s) for s, _ in finished)
    assert total >= 500
    assert st["league"][SELF]["games"] == 0
    assert st["opp_games"] == st["games"]
    assert 0.2 < total / st["decisions"] < 0.75  # only the learner seat is stored


def test_pfsp_prefers_harder_opponent():
    lg = League(0.0, 0.0, 6, np.random.default_rng(0))
    lg.add_snapshot("snap:A", object())
    lg.add_snapshot("snap:B", object())
    for _ in range(200):
        lg.record("snap:A", 1.0)  # learner always beats A
        lg.record("snap:B", 0.0)  # learner always loses to B
    assert lg.weight_of("snap:B") > lg.weight_of("snap:A")
    counts = {"snap:A": 0, "snap:B": 0, BASE: 0}
    for _ in range(2000):
        counts[lg.sample()] += 1
    assert counts["snap:B"] > counts["snap:A"]


def test_snapshot_eviction_keeps_newest():
    lg = League(0.4, 0.15, 2, np.random.default_rng(0))
    for k in range(4):
        lg.add_snapshot(f"snap:{k}", object())
        lg.record(f"snap:{k}", 1.0)
    assert lg.names() == [BASE, "snap:2", "snap:3"]
    assert "snap:0" not in lg.ema and "snap:1" not in lg.ema


def test_ygo_state_cleared_on_episode_end(engine):
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    col, _, _, ygo = _league_collector(engine, decks, seed=2, p_self=0.0, p_ygo=1.0)
    _, st = col.collect(300)
    assert st["league"][YGO]["games"] > 0
    # one reset per finished ygo episode, each token reset once, and tokens
    # rotate (more episodes than envs, so keys are never reused)
    assert len(ygo.resets) >= st["league"][YGO]["games"]
    assert len(set(ygo.resets)) == len(ygo.resets)
    assert len(set(ygo.seen_keys)) > 8


def test_reset_keys_drops_only_named_games():
    from decklight.eval.ygoagent import YgoAgentPolicy

    pol = YgoAgentPolicy.__new__(YgoAgentPolicy)  # no TFLite needed: only the state table
    pol._states = {(5, 0): ("h5", "c5"), (6, 1): ("h6", "c6")}
    pol.reset_keys([5])
    assert pol._states == {(6, 1): ("h6", "c6")}


class RecStubYgo:
    """YGO stand-in that records the ga flag of every row it plays."""

    def __init__(self):
        self.calls = 0
        self.seen_ga: list[int] = []
        self.resets: list[int] = []

    def __call__(self, buf, rows, keys=None):
        assert keys is not None and len(keys) == len(np.asarray(rows))
        self.calls += 1
        for i in np.asarray(rows).tolist():
            self.seen_ga.append(int(buf["ga"][int(i)]))
        return np.zeros(len(np.asarray(rows)), np.int32)

    def reset_keys(self, game_ids):
        self.resets.extend(int(g) for g in game_ids)


def test_rule_mixed_never_trains_gauntlet(engine):
    """Overfitting rule: lseat is the pool seat, the gauntlet seat is never stored,
    YGO only plays gauntlet episodes on the gauntlet seat (tiny d=16 nets)."""
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    n = len(decks)
    assert n >= 4
    n_pool, ga_start, ga_count = n - 2, n - 2, 2
    dev = torch.device("cpu")
    model, ref = _tiny(decks, True), _tiny(decks, False)
    league = League(0.3, 0.0, 0, np.random.default_rng(41), ga_ygo_share=0.5)
    ygo = RecStubYgo()
    vec = VecEnv(engine, decks, 8, 2, seed=7)
    vec.set_max_decisions(60)
    vec.set_pool_count(n_pool)
    vec.set_ga_share(0.5, ga_start, ga_count, mixed=True)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False))
    col = Collector(vec, model, ref, store, 0, False, dev, 0.3, -0.5, 2,
                    np.random.default_rng(7), "fp32", league, ygo,
                    ga_share=0.5, ga_range=(ga_start, ga_count), ga_mixed=True)
    finished, st = col.collect(500)
    assert st.get("ga_games", 0) > 0, st
    assert ygo.calls > 0, "no YGO rows over 500 samples at ga_share 0.5"
    assert all(g != 0 for g in ygo.seen_ga), "YGO played a non-gauntlet episode"
    assert col.ygo_bad == 0
    assert st.get("learner_ga", -1) == 0, st
    # In-flight episodes: lseat is the non-gauntlet seat with exactly one gauntlet deck.
    for i in range(vec.num_envs):
        f = int(col.ep_ga[i])
        if f in (1, 2):
            assert int(col.lseat[i]) == 2 - f, (i, f, int(col.lseat[i]))
            d0, d1 = int(col.ep_d0[i]), int(col.ep_d1[i])
            assert (ga_start <= d0 < ga_start + ga_count) != (ga_start <= d1 < ga_start + ga_count), (d0, d1)


def test_league_fail_checks():
    tr = _train_rl_mod()
    lg = League(0.4, 0.15, 6, np.random.default_rng(0))
    calls: list[str] = []
    tr.league_fail_checks(3, {"opp_games": 2, "opp_learner_samples": 0}, lg, 40, 6, calls.append)
    assert len(calls) == 1 and "0 learner samples" in calls[0]
    tr.league_fail_checks(80, {"opp_games": 0, "opp_learner_samples": 0}, lg, 40, 6, calls.append)
    assert len(calls) == 2 and "none added" in calls[1]
    lg.add_snapshot("snap:80", object())
    tr.league_fail_checks(80, {"opp_games": 1, "opp_learner_samples": 5}, lg, 40, 6, calls.append)
    assert len(calls) == 2
