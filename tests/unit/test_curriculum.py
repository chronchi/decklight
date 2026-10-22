"""Pass-rate curriculum: tracker math, C++ weighted draws, Collector start indices."""

import numpy as np
import pytest
import torch

from decklight.decks import load_ydk
from decklight.engine.cpp_env import CppEnv, VecEnv
from decklight.engine.duel import DuelConfig
from decklight.model.policy import PolicyNet
from decklight.rl.curriculum import PassRate
from decklight.rl.ppo import SlotStore, gather
from decklight.rl.rollout import Collector

from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"


def test_weights_prior_and_eps():
    cur = PassRate(3, eps=0.02, min_n=4)
    # Unseen: Beta(1,1) prior p=0.5 -> 0.25 + eps (explored, never zero).
    assert cur.p_hat(0) == 0.5
    assert cur.weights() == [0.27, 0.27, 0.27]
    assert cur.stats() == {"starts": 3, "seen": 0, "trainable": 0, "solved": 0,
                           "hopeless": 0, "ess_share": 1.0}


def test_trainable_peaks_at_half():
    cur = PassRate(2, eps=0.02, min_n=4)
    for _ in range(3):
        cur.record(0, 1.0)
        cur.record(0, 0.0)  # 3-3: p-hat 0.5 -> max variance weight
    for _ in range(5):
        cur.record(1, 1.0)
    cur.record(1, 0.0)  # 6-1 record... p-hat = 7/9
    assert cur.weight(0) > cur.weight(1) > 0.02
    assert cur.stats()["trainable"] == 2


def test_solved_and_hopeless_get_eps_only():
    cur = PassRate(3, eps=0.02, min_n=4)
    for _ in range(4):
        cur.record(0, 1.0)  # solved K/K
        cur.record(1, 0.0)  # hopeless 0/K
        cur.record(2, 1.0 if _ % 2 else 0.0)  # 2-2: trainable
    # start 2 has n=4, wins=2 -> trainable
    assert cur.is_solved(0) and cur.is_solved(1) and not cur.is_solved(2)
    assert cur.weight(0) == 0.02 and cur.weight(1) == 0.02
    assert cur.weight(2) > 0.02
    assert cur.solved_mask([0, 1, 2, -1]) == [True, True, False, False]
    s = cur.stats()
    assert (s["seen"], s["trainable"], s["solved"], s["hopeless"]) == (3, 1, 1, 1)


def test_min_n_guards_small_samples():
    cur = PassRate(1, eps=0.02, min_n=4)
    for _ in range(3):
        cur.record(0, 1.0)  # 3/3 but n < min_n: not solved, keeps variance weight
    assert not cur.is_solved(0)
    assert cur.weight(0) == pytest.approx((4 / 5) * (1 / 5) + 0.02)


def test_hint_bonus_and_roundtrip(tmp_path):
    cur = PassRate(2, eps=0.0, min_n=4, hint_bonus={1: 0.5})
    assert cur.weight(1) == 0.25 + 0.5
    p = tmp_path / "curriculum.json"
    cur.record(0, 1.0)
    cur.save(p)
    back = PassRate.load(p)
    assert back.weights() == cur.weights() and back.n == cur.n and back.wins == cur.wins
    assert back.hint_bonus == {1: 0.5}


def test_record_rejects_bad_input():
    cur = PassRate(2)
    with pytest.raises(ValueError):
        cur.record(2, 1.0)
    with pytest.raises(ValueError):
        cur.record(-1, 1.0)
    with pytest.raises(ValueError):
        cur.record(0, 0.7)


def _starts(engine, decks):
    out = []
    for seed in (42, 43):
        cfg = DuelConfig(decks[0], decks[1], seed=seed)
        env = CppEnv(engine, cfg)
        assert env.reset() is not None
        for _ in range(3):
            assert env.step(0) is not None
        out.append((cfg.main_of(0), list(decks[0].extra), cfg.main_of(1),
                    list(decks[1].extra), seed, [0] * 3))
    return out


def _ref_obs(engine, decks, seed):
    cfg = DuelConfig(decks[0], decks[1], seed=seed)
    env = CppEnv(engine, cfg)
    env.reset()
    for _ in range(3):
        ob = env.step(0)
    return {k: np.array(ob[k]) for k in ("cards", "global", "action_mask")}


def test_weighted_draws_match_weights_and_start_matches(engine):
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    starts = _starts(engine, decks)
    refs = [_ref_obs(engine, decks, s) for s in (42, 43)]
    vec = VecEnv(engine, decks, 8, 2, seed=11)
    vec.set_starts(starts, 1.0, 99)
    vec.set_start_weights([3.0, 1.0])
    counts = [0, 0]
    for _ in range(250):
        vec.reset()
        got = np.asarray(vec.buf["start"])
        assert set(got.tolist()) <= {0, 1}
        for i in range(8):
            counts[int(got[i])] += 1
    # 75/25 expected of 2000 draws: +/-3 pts = [1440, 1560] / [440, 560].
    assert 1440 <= counts[0] <= 1560, counts
    assert 440 <= counts[1] <= 560, counts
    # buf["start"] matches the drawn start: obs equals that start's prefix replay.
    vec.reset()
    got = np.asarray(vec.buf["start"])
    for i in range(8):
        ref = refs[int(got[i])]
        for k in ref:
            assert np.array_equal(vec.buf[k][i], ref[k]), (i, k)


def test_zero_weight_never_drawn_and_no_call_is_deterministic(engine):
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    starts = _starts(engine, decks)
    vec = VecEnv(engine, decks, 4, 2, seed=5)
    vec.set_starts(starts, 1.0, 7)
    vec.set_start_weights([1.0, 0.0])
    for _ in range(10):
        vec.reset()
        assert (np.asarray(vec.buf["start"]) == 0).all()
    # No set_start_weights call: uniform draws, same seed -> identical sequences.
    seqs = []
    for _ in range(2):
        v = VecEnv(engine, decks, 4, 2, seed=5)
        v.set_starts(starts, 1.0, 7)
        seq = []
        for _ in range(10):
            v.reset()
            seq.append(tuple(np.asarray(v.buf["start"]).tolist()))
        seqs.append(seq)
    assert seqs[0] == seqs[1]
    flat = [x for s in seqs[0] for x in s]
    assert set(flat) == {0, 1}  # uniform: both drawn


def _tiny_net(decks, value: bool):
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    return PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, value=value).eval()


def test_collector_reports_start_per_trajectory(engine):
    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    starts = _starts(engine, decks)
    dev = torch.device("cpu")
    model, ref = _tiny_net(decks, True), _tiny_net(decks, False)
    vec = VecEnv(engine, decks, 8, 2, seed=3)
    vec.set_max_decisions(60)
    vec.set_starts(starts, 1.0, 3 + 0xDA65)
    store = SlotStore(8000, gather(vec.buf, np.arange(1), 0, False))
    col = Collector(vec, model, ref, store, 0, False, dev, 0.0, -0.5, 2,
                    np.random.default_rng(3), "fp32")
    finished, st = col.collect(300)
    assert len(st["fin_starts"]) == len(finished)
    assert all(f >= 0 for f in st["fin_starts"])  # p=1: every episode is a DAGS episode
    # The tracker consumes the report; the push reaches the VecEnv.
    # slots are (start, seat): self-play records both seats of a game, one win and one loss, so a
    # per-start key would pin every start at p = 0.5 (never solved/hopeless)
    cur = PassRate(2 * len(starts))
    for (_, rew), fst in zip(finished, st["fin_starts"]):
        cur.record(int(fst), 1.0 if rew > 0 else (0.5 if rew == 0 else 0.0))
    assert cur.stats()["seen"] >= 2
    seats = {int(f) % 2 for f in st["fin_starts"]}
    assert seats == {0, 1}  # self-play: both seats of each start are tracked separately
    assert len(cur.start_weights()) == len(starts)
    vec.set_start_weights(cur.start_weights())  # no raise: len matches, sum > 0


def test_start_weights_take_the_harder_seat():
    """A start where seat 0 always wins (solved) and seat 1 is mixed keeps seat 1's weight."""
    cur = PassRate(4, eps=0.02, min_n=4)
    for _ in range(4):
        cur.record(0, 1.0)  # start 0, seat 0: solved
    for r in (1.0, 0.0, 1.0, 0.0):
        cur.record(1, r)  # start 0, seat 1: trainable
    w = cur.start_weights()
    assert len(w) == 2
    assert w[0] == cur.weight(1) > cur.weight(0) == 0.02
    assert cur.solved_mask([0, 1, -1]) == [True, False, False]
