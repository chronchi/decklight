"""Combo-line opponents (rl/lines.py): fingerprints, script tracking, VecEnv replay and the Collector path (CPU, tiny)."""

from pathlib import Path

import numpy as np
import torch

from decklight.decks import load_ydk
from decklight.engine.cpp_env import CppEnv, VecEnv
from decklight.engine.duel import DuelConfig
from decklight.rl import dags
from decklight.rl import lines as L
from decklight.rl.ppo import SlotStore, gather
from decklight.rl.rollout import Collector

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"


def _decks():
    return [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]


def _record(engine, decks, seed=42, k=6, n=40):
    """A turn record shaped like plan_turns output: choices (last legal option each time), labels, actors."""
    cfg = DuelConfig(decks[0], decks[1], seed=seed)
    env = CppEnv(engine, cfg)
    obs, choices, actors = env.reset(), [], []
    while obs is not None and len(choices) < n:
        actors.append(env.player)
        c = int(np.flatnonzero(np.asarray(obs["action_mask"]).astype(bool))[-1])
        choices.append(c)
        obs = env.step(c)
    conf = {"deck0_main": list(decks[0].main), "deck0_extra": list(decks[0].extra),
            "deck1_main": list(decks[1].main), "deck1_extra": list(decks[1].extra),
            "seed": seed, "hand": 5, "draw": 1, "setup": [], "shuffle": True}
    return {"line_id": "t/line", "config": conf, "auto_single": True, "choices": choices, "actors": actors,
            "labels": ["prefix"] * k + ["0"] * (len(choices) - k)}


def test_fingerprint_uses_legal_rows_only():
    a = np.arange(40).reshape(4, 10)
    m = np.array([1, 1, 0, 0])
    b = a.copy()
    b[3] = 99  # an illegal row changes nothing
    assert L.fingerprint(a, m) == L.fingerprint(b, m)
    b[1, 0] = -1
    assert L.fingerprint(a, m) != L.fingerprint(b, m)
    assert L.fingerprint(a.astype(np.int16), m) == L.fingerprint(a, m)


def test_tracker_follow_break_complete():
    a = np.arange(20).reshape(2, 10)
    m = np.array([1, 1])
    fp = L.fingerprint(a, m)
    tr = L.LineTracker(2)
    tr.begin(0, {"script": [(fp, 1), (fp, 0)]})
    tr.begin(1, {"script": [(fp + 1, 1)]})
    assert tr.next(0, a, m) == 1 and tr.next(0, a, m) == 0
    assert tr.next(0, a, m) is None and tr.completed == 1
    assert tr.next(1, a, m) is None and tr.broken == 1 and not tr.on[1]
    tr.begin(1, None)
    assert tr.next(1, a, m) is None and tr.followed == 2


def test_second_start_replays_in_vecenv(engine):
    decks = _decks()
    rec = _record(engine, decks)
    st = L.second_start(engine, rec)
    opp = rec["actors"][6]
    assert st["our_seat"] == 1 - opp and st["prefix"] == rec["choices"][:6]
    assert len(st["script"]) == sum(a == opp for a in rec["actors"][6:])
    vec = VecEnv(engine, decks, 1, 1, seed=0)
    vec.set_starts([dags.to_set_starts(st)], 1.0, 0)
    buf = vec.reset()
    tr = L.LineTracker(1)
    tr.begin(0, {"script": [tuple(x) for x in st["script"]]})
    for c, actor in zip(rec["choices"][6:], rec["actors"][6:]):
        assert int(buf["player"][0]) == actor
        if actor == opp:
            assert tr.next(0, buf["actions"][0], buf["action_mask"][0]) == c
        buf = vec.step(np.array([c], np.int32))
    assert tr.completed == 1 and tr.broken == 0


def _tiny_net(decks, value: bool):
    from decklight.model.policy import PolicyNet

    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    return PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, value=value).eval()


def test_collector_line_episodes(engine, tmp_path):
    decks = _decks()
    rec = _record(engine, decks)
    st = L.second_start(engine, rec)
    f = tmp_path / "lines.jsonl"
    import json

    f.write_text(json.dumps(st) + "\n")
    starts, meta = L.load_line_starts(f)
    vec = VecEnv(engine, decks, 4, 2, seed=0)
    vec.set_max_decisions(60)
    vec.set_starts(starts, 1.0, 0xDA65)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False))
    col = Collector(vec, _tiny_net(decks, True), _tiny_net(decks, False), store, 0, False, torch.device("cpu"),
                    0.0, -0.5, 2, np.random.default_rng(0), "fp32", line_meta=meta)
    _, s = col.collect(100)
    assert s["line_games"] == s["games"] > 0
    assert s["line_followed"] > 0
    assert s["line_broken"] + s["line_completed"] > 0
    assert (col.lseat == meta[0]["our_seat"]).all()
