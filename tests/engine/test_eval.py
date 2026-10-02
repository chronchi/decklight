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
