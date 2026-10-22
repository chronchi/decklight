"""DAGS mid-game starts: C++ set_starts, builder helpers and Collector stats (CPU, tiny)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from decklight.decks import load_ydk
from decklight.engine.cpp_env import CppEnv, VecEnv
from decklight.engine.duel import DuelConfig
from decklight.model.policy import PolicyNet
from decklight.rl import dags
from decklight.rl.ppo import SlotStore, gather
from decklight.rl.rollout import Collector

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"
ROOT = Path(__file__).resolve().parents[2]

_BUILDER = None


def _builder_mod():
    global _BUILDER
    if _BUILDER is None:
        p = ROOT / "scripts" / "build_dags_starts.py"
        spec = importlib.util.spec_from_file_location("build_dags_starts_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BUILDER = mod
    return _BUILDER


def _decks():
    return [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]


def _start_with_engine(engine, decks, seed=42, n=3):
    cfg = DuelConfig(decks[0], decks[1], seed=seed)
    env = CppEnv(engine, cfg)
    assert env.reset() is not None
    for _ in range(n):
        assert env.step(0) is not None
    return ((cfg.main_of(0), list(decks[0].extra), cfg.main_of(1), list(decks[1].extra),
             seed, [0] * n))


def _ref_obs(engine, decks, seed=42, n=3):
    cfg = DuelConfig(decks[0], decks[1], seed=seed)
    env = CppEnv(engine, cfg)
    env.reset()
    for _ in range(n):
        ob = env.step(0)
    return {k: np.array(ob[k]) for k in ("cards", "global", "action_mask")}


def _snap(vec):
    return {k: np.array(vec.buf[k]) for k in ("cards", "global", "action_mask")}


def test_set_starts_p0_bit_identical(engine):
    decks = _decks()
    starts = [_start_with_engine(engine, decks)]
    a = VecEnv(engine, decks, 4, 2, seed=7)
    a.reset()
    b = VecEnv(engine, decks, 4, 2, seed=7)
    b.set_starts(starts, 0.0, 123)
    b.reset()
    sa, sb = _snap(a), _snap(b)
    for k in sa:
        assert np.array_equal(sa[k], sb[k]), k
    assert not b.buf["dags"].any() and b.dags_used == 0 and b.dags_bad == 0
    acts = np.zeros(4, np.int32)
    for _ in range(5):
        a.step(acts)
        b.step(acts)
    sa, sb = _snap(a), _snap(b)
    for k in sa:
        assert np.array_equal(sa[k], sb[k]), k


def test_p1_prefix_matches_cppenv_replay(engine):
    decks = _decks()
    starts = [_start_with_engine(engine, decks)]
    ref = _ref_obs(engine, decks)
    vec = VecEnv(engine, decks, 3, 2, seed=1)
    vec.set_starts(starts, 1.0, 99)
    vec.reset()
    assert bool(vec.buf["dags"].all())
    assert vec.dags_used == 3 and vec.dags_bad == 0
    for i in range(3):
        for k in ref:
            assert np.array_equal(vec.buf[k][i], ref[k]), (i, k)


def test_bad_prefix_falls_back(engine):
    decks = _decks()
    cfg = DuelConfig(decks[0], decks[1], seed=42)
    bad = (cfg.main_of(0), list(decks[0].extra), cfg.main_of(1), list(decks[1].extra),
           42, [999999])
    vec = VecEnv(engine, decks, 2, 2, seed=1)
    vec.set_starts([bad], 1.0, 99)
    vec.reset()
    assert not vec.buf["dags"].any()
    assert vec.dags_used == 0 and vec.dags_bad == 2
    assert (np.asarray(vec.buf["player"]) != -1).all()
    assert np.asarray(vec.buf["action_mask"]).sum() > 0


def _tiny_record(decks, seed=5, n=12):
    """A synthetic demo record with a synthetic `turns` column."""
    cfg = DuelConfig(decks[0], decks[1], seed=seed)
    return {
        "game_id": "tiny-1",
        "decks": ["BlueEyes", "Branded"],
        "config": {
            "deck0_main": list(decks[0].main), "deck0_extra": list(decks[0].extra),
            "deck1_main": list(decks[1].main), "deck1_extra": list(decks[1].extra),
            "seed": seed, "hand": 5, "draw": 1, "setup": [],
        },
        "auto_single": True,
        "choices": [0] * n,
        "turns": [1] * 4 + [2] * 4 + [3] * 4,
    }


def test_builder_helpers_holdout_cuts_replay(engine):
    decks = _decks()
    rec = _tiny_record(decks)
    hold = {dags.deck_key(decks[0].main, decks[0].extra)}
    assert dags.is_holdout_game(rec, hold)
    rec2 = dict(rec, config=dict(rec["config"], deck0_main=list(decks[2].main),
                                 deck0_extra=list(decks[2].extra)))
    if dags.deck_key(decks[2].main, decks[2].extra) != dags.deck_key(decks[0].main, decks[0].extra):
        assert not dags.is_holdout_game(rec2, hold)
    rng = np.random.default_rng(0)
    cuts = dags.cut_points(12, rec["turns"], rng)
    assert 4 in cuts and 8 in cuts  # turn boundaries
    assert cuts == sorted(set(cuts)) and all(1 <= k < 12 for k in cuts)
    assert 2 <= len(cuts) <= 4  # 2 turn boundaries + up to 2 random (may collide)
    # Every emitted start replays through the real builder path.
    mod = _builder_mod()
    mod._init()
    idx, first_dead, err = mod._replay((0, rec, "tiny"))
    assert idx == 0 and err == "" and first_dead >= max(cuts)
    for k in cuts:
        assert k <= first_dead, (k, first_dead, err)
        line = dags.start_record(rec, k, "tiny")
    assert line["turn"] == rec["turns"][cuts[-1]] and line["prefix"] == [0] * cuts[-1]
    assert dags.cut_points(1, [1], rng) == []


def test_load_starts_roundtrip(engine, tmp_path):
    decks = _decks()
    rec = _tiny_record(decks)
    lines = [dags.start_record(rec, k, "tiny") for k in (4, 8)]
    p = tmp_path / "starts.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    starts = dags.load_starts(p)
    assert len(starts) == 2 and len(starts[0]) == 10
    vec = VecEnv(engine, decks, 2, 2, seed=1)
    vec.set_starts(starts, 1.0, 9)  # accepted by the binding
    vec.reset()
    assert bool(vec.buf["dags"].all())


def _tiny_net(decks, value: bool):
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    return PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, value=value).eval()


def _collector(engine, decks, starts, p, seed=0):
    dev = torch.device("cpu")
    model, ref = _tiny_net(decks, True), _tiny_net(decks, False)
    vec = VecEnv(engine, decks, 8, 2, seed=seed)
    vec.set_max_decisions(60)
    if starts is not None:
        vec.set_starts(starts, p, seed + 0xDA65)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False))
    col = Collector(vec, model, ref, store, 0, False, dev, 0.0, -0.5, 2,
                    np.random.default_rng(seed), "fp32")
    return col


def test_collector_dags_stats(engine):
    decks = _decks()
    starts = [_start_with_engine(engine, decks)]
    col = _collector(engine, decks, starts, 1.0)
    finished, st = col.collect(300)
    assert sum(len(s) for s, _ in finished) >= 300
    assert st["dags_games"] == st["games"] > 0  # p=1: every episode is a DAGS episode
    assert st["dags_bad"] == 0
    assert 0.0 <= st["dags_wr"] <= 1.0 and np.isfinite(st["dags_wr"])
    col0 = _collector(engine, decks, None, 0.0, seed=3)
    _, st0 = col0.collect(300)
    assert st0["dags_games"] == 0 and st0["dags_bad"] == 0
    colm = _collector(engine, decks, starts, 0.5, seed=5)
    _, stm = colm.collect(300)
    assert 0 < stm["dags_games"] < stm["games"]  # both episode kinds in one run
    assert set(stm) >= {"dags_games", "dags_wr", "dags_bad"}
