"""Match record v1 (ADR 0010): schema-valid, replays from raw responses, rejects tampered records."""

import copy
import random

import pytest

from decklight.engine.duel import DuelConfig
from decklight.engine.env import PyEnv
from decklight.play import match_record as MR

pytest.importorskip("jsonschema")


def _play(engine, d0, d1, seed, rng, max_steps=4000):
    env = PyEnv(engine, DuelConfig(d0, d1, seed=seed), auto_single=False)
    env.reset()
    actors, opts, cur = [], [], []
    while not env.done and max_steps:
        max_steps -= 1
        n = len(env.duel.responses)
        player = env.player
        i = rng.randrange(len(env.opts))
        env.step(i)
        cur.append(i)
        if len(env.duel.responses) > n:  # this pick completed a core response
            actors.append(player)
            opts.append(cur)
            cur = []
    return env, actors, opts


@pytest.fixture(scope="module")
def game(engine, decks):
    env, actors, opts = _play(engine, decks["BlueEyes"], decks["Branded"], 4242, random.Random(7))
    assert env.done and env.duel.winner in (0, 1), "the test game must finish with a winner"
    seats = [
        {"type": "human", "player": "test-install"},
        {"type": "bot", "bot_mode": "random", "model": "random"},
    ]
    rec = MR.from_duel(
        env.duel, seats=seats, mode="practice", app="decklight-test/0", actors=actors, opts=opts
    )
    return env, rec


def test_record_is_schema_valid(game):
    _, rec = game
    assert MR.validate(rec) == []
    assert rec["kind"] == "full" and rec["responses"], "empty record"


def test_record_replays_to_the_same_result(game, engine):
    env, rec = game
    assert MR.check(rec, engine) == []
    duel = MR.replay(rec, engine)
    assert duel.trace == env.duel.trace  # the whole message stream, not just the winner


def test_tampered_record_is_rejected(game, engine):
    _, rec = game
    bad = copy.deepcopy(rec)
    bad["result"]["winner"] = 1 - bad["result"]["winner"] if bad["result"]["winner"] in (0, 1) else 0
    assert MR.check(bad, engine)
    short = copy.deepcopy(rec)
    short["responses"] = short["responses"][: len(short["responses"]) // 2]
    assert MR.check(short, engine)
    seed = copy.deepcopy(rec)
    seed["engine_seed"][0] = str((int(seed["engine_seed"][0]) + 1) % 2**64)
    assert MR.check(seed, engine)  # another seed deals other hands: the responses no longer fit


def test_save_load_and_shareable(game, tmp_path):
    _, rec = game
    path = MR.save(rec, tmp_path)
    assert MR.load(path) == rec
    shared = MR.shareable({**rec, "transcript": "x", "events": []})
    assert "transcript" not in shared and "events" not in shared
    assert all("name" not in d for d in shared["decks"])
    assert MR.validate(shared) == []


def test_uuid7_is_time_ordered():
    a, b = MR.uuid7(1_000), MR.uuid7(2_000)
    assert a < b and a[14] == "7" and a[19] in "89ab"
