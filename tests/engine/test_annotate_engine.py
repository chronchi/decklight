"""End-to-end annotation of a short random game (engine replay, no model)."""

import random

from ygo_zero.analysis import annotate as A
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig

MAX_STEPS = 60


def _roll_short_game(engine, decks, seed, max_steps=MAX_STEPS):
    names = sorted(decks)
    env = CppEnv(engine, DuelConfig(decks[names[0]], decks[names[1]], seed=seed))
    obs = env.reset()
    rng = random.Random(seed)
    choices, first = [], None
    for k in range(max_steps):
        if obs is None:
            break
        n = int(obs["action_mask"].sum())
        ch = rng.randrange(n)
        if first is None:
            first = (n, ch, int(env.player), int(obs["global"][2]))
        choices.append(ch)
        obs = env.step(ch)
    return names, choices, first


def _describe_stub(env, obs, i, names):
    return f"option {i}"


def test_annotate_random_game_end_to_end(engine, decks):
    names, choices, first = _roll_short_game(engine, decks, seed=123)
    assert len(choices) >= 4
    n0, ch0, seat0, turn0 = first
    cands = [ch0, (ch0 + 1) % n0] if n0 >= 2 else [ch0]
    rec = {
        "game_idx": 0,
        "decks": names[:2],
        "seed": 123,
        "s_seat": seat0,
        "s_first": seat0 == 0,
        "deck_s": names[seat0],
        "deck_r": names[1 - seat0],
        "winner": None,
        "truncated": True,
        "turns": 3,
        "choices": choices,
        "search": [
            {
                "k": 0,
                "seat": seat0,
                "kind": "own",
                "turn": turn0,
                "cands": cands,
                "samples": [[0.3, 0.25]] * len(cands) if len(cands) == 1 else [[0.3, 0.25], [0.1, 0.05]],
                "chosen": cands[0],
                "played": cands[-1],
            }
        ],
    }
    snaps, end_snap, descs = A.replay(rec, engine, lambda n: decks[n], _describe_stub, {})
    assert len(snaps) >= 4
    ann = A.annotate_game(rec, snaps, end_snap, descs)
    assert ann["summary"]["n_searched"] == 1
    assert ann["curve"][turn0] is not None
    md = A.render_game_md(ann)
    assert f"Game {rec['game_idx']}" in md
    sm = A.summarize([ann], ["test"], "test")
    assert sm["meta"]["n_games"] == 1
    A.render_summary_md(sm)


def test_completed_game_winner_check(engine, decks):
    # A finished random game must replay exactly; a flipped winner must fail loud.
    found = None
    for seed in (123, 124, 125, 126):
        nms, chs, fst = _roll_short_game(engine, decks, seed, max_steps=500)
        env = CppEnv(engine, DuelConfig(decks[nms[0]], decks[nms[1]], seed=seed))
        obs = env.reset()
        for ch in chs:
            if obs is None:
                break
            obs = env.step(ch)
        if obs is None:
            found = (seed, nms, chs, fst, int(env.winner))
            break
    if found is None:
        import pytest

        pytest.skip("no seed finished a game within the step cap")
    seed, names, choices, first, winner = found
    (_n0, _ch0, seat0, _turn0) = first
    base = {
        "game_idx": 0,
        "decks": names[:2],
        "seed": seed,
        "s_seat": seat0,
        "s_first": seat0 == 0,
        "deck_s": names[seat0],
        "deck_r": names[1 - seat0],
        "truncated": False,
        "turns": 3,
        "choices": choices,
        "search": [],
    }
    import pytest

    ok = dict(base, winner=winner)
    snaps, _end_snap, _descs = A.replay(ok, engine, lambda n: decks[n], _describe_stub, {})
    assert len(snaps) == len(choices)
    bad = dict(base, winner=1 - winner)
    with pytest.raises(A.ReplayMismatch):
        A.replay(bad, engine, lambda n: decks[n], _describe_stub, {})
