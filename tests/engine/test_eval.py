import numpy as np
import pytest

pytest.importorskip("ygo_zero.engine.cpp_env")

from ygo_zero.eval.match import play_match, wilson
from ygo_zero.eval.policies import GreedyPolicy, RandomPolicy


def test_schedule_plays_each_game_once(engine, decks):
    from ygo_zero.engine.cpp_env import VecEnv

    names = sorted(decks)
    sched = [(0, 1, 5), (1, 0, 5), (2, 3, 6), (3, 3, 7), (1, 2, 8)]
    vec = VecEnv(engine, [decks[n] for n in names], num_envs=3, num_threads=2, seed=0)
    vec.set_schedule(sched)
    buf, pol, eps = vec.reset(), RandomPolicy(0), []
    while vec.remaining > 0:
        live = (buf["player"] >= 0).nonzero()[0]
        actions = buf["player"].copy() * 0
        actions[live] = pol(buf, live)
        vec.step(actions)
        eps += vec.pop_finished()
    assert sorted(e.index for e in eps) == list(range(len(sched)))
    assert all((e.deck0, e.deck1, e.seed) == sched[e.index] for e in eps)
    assert (buf["player"] == -1).all() and not buf["action_mask"].any()  # all idle


def test_match_seat_swap(engine, decks):
    names = sorted(decks)
    games = play_match(engine, [decks[n] for n in names], GreedyPolicy(1), RandomPolicy(2),
                       pairs=[(0, 1)], seeds=range(3), num_envs=4, num_threads=2)  # fmt: skip
    assert len(games) == 6 and sum(g.a_first for g in games) == 3
    assert all(g.winner_is_a in (0.0, 0.5, 1.0) for g in games)


def test_wilson():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-9


class _FirstLegal:
    """Pure function of the observation row (no rng): the only policies in eval/policies.py
    (greedy/random) both break ties with an rng draw, whose sequence depends on call grouping,
    so sync-vs-async equality needs these instead."""

    name = "first-legal"

    def __call__(self, buf, rows):
        return np.argmax(buf["action_mask"][rows].astype(np.int8), axis=1).astype(np.int32)


class _LastLegal:
    name = "last-legal"

    def __call__(self, buf, rows):
        m = buf["action_mask"][rows][:, ::-1]
        return (m.shape[1] - 1 - np.argmax(m.astype(np.int8), axis=1)).astype(np.int32)


def test_play_match_async_equals_sync(engine, decks):
    names = sorted(decks)[:4]
    deck_list = [decks[n] for n in names]
    pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
    kw = dict(pairs=pairs, seeds=[0, 1], num_envs=6, num_threads=3, max_decisions=400)
    sync = play_match(engine, deck_list, _FirstLegal(), _LastLegal(), mode="sync", **kw)
    async_ = play_match(engine, deck_list, _FirstLegal(), _LastLegal(), mode="async", **kw)
    assert async_ == sync
    assert len(sync) == 2 * len(pairs) * 2  # seat swap × seeds


def test_play_match_async_no_idle_send(engine, decks):
    names = sorted(decks)[:4]
    deck_list = [decks[n] for n in names]
    pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
    seeds = [0, 1]
    asked = []

    class Recorder:
        name = "recorder"

        def __init__(self, inner):
            self.inner = inner

        def __call__(self, buf, rows):
            assert rows.dtype == np.int64
            for r in (int(r) for r in rows):
                assert int(buf["player"][r]) != -1, f"policy asked about idle row {r}"
                asked.append(r)
            return self.inner(buf, rows)

    games = play_match(engine, deck_list, Recorder(GreedyPolicy(1)), Recorder(RandomPolicy(2)),
                       pairs=pairs, seeds=seeds, num_envs=6, num_threads=3,
                       max_decisions=400, mode="async")  # fmt: skip
    assert asked, "policies were never called"
    expected = sorted((da, db, s, first) for da, db in pairs for s in seeds for first in (True, False))
    got = sorted((g.deck_a, g.deck_b, g.seed, g.a_first) for g in games)
    assert got == expected  # every scheduled game exactly once
