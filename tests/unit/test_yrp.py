""".yrpX (EDOPro/Multirole replay) → match record v1: round trip through a Multirole-layout writer, both card-add
orders recovered, tampered replays rejected, malformed bytes raise ValueError."""

import random

import pytest

from decklight.engine import constants as C
from decklight.engine.duel import DuelConfig, seed_words
from decklight.engine.env import PyEnv
from decklight.play import match_record as MR
from decklight.play import yrp as Y

pytest.importorskip("jsonschema")


def _game(engine, d0, d1, seed, order):
    env = PyEnv(engine, DuelConfig(d0, d1, seed=seed, team_order=order), auto_single=False)
    env.reset()
    rng = random.Random(seed)
    actors = []
    while not env.done:
        n, p = len(env.duel.responses), env.player
        env.step(rng.randrange(len(env.opts)))
        if len(env.duel.responses) > n:
            actors.append(p)
    assert env.duel.winner in (0, 1)
    duel = env.duel
    cfg = duel.config
    y = Y.Yrp(
        seed=tuple(seed_words(seed)),
        names=[["alice"], ["bob"]],
        lp=cfg.lp,
        hand=cfg.hand,
        draw=cfg.draw,
        flags=cfg.flags,
        decks=Y.yrp_decks(cfg),
        extra_cards=[],
        responses=list(duel.responses),
    )
    return duel, actors, y


@pytest.fixture(scope="module", params=[(0, 1), (1, 0)], ids=["order01", "order10"])
def replay(request, engine, decks):
    duel, actors, y = _game(engine, decks["Labrynth"], decks["SkyStrikerAce"], 9001, request.param)
    return request.param, duel, actors, y, Y.write_yrpx(y, duel.trace, ts=1_790_000_000)


def test_parse_round_trip(replay):
    _, duel, _, y, data = replay
    x = Y.parse_yrpx(data)
    assert x.yrp == y
    assert Y.core_messages(x) == Y.recorded(duel.trace)
    assert x.names == [["alice"], ["bob"]]
    assert Y.parse_yrpx(Y.write_yrpx(y, duel.trace, compress=False)).yrp == y


def test_to_record_finds_the_add_order_and_replays(replay, engine):
    order, duel, actors, _, data = replay
    rec, problems = Y.to_record(Y.parse_yrpx(data), engine, viewer=0, app="decklight-test/0")
    assert problems == [] and rec is not None
    assert tuple(rec.get("add_order", (0, 1))) == order
    assert [e["seat"] for e in rec["responses"]] == actors
    assert rec["result"]["winner"] == duel.winner and rec["result"]["turns"] == duel.turn
    assert rec["source"] == "yrpx" and rec["cards"]["server"] is True
    assert MR.validate(rec) == []
    assert MR.check(rec, engine) == []


def test_tampered_replay_is_rejected(replay, engine):
    _, duel, _, y, _ = replay
    bad = Y.Yrp(**{**y.__dict__, "responses": list(y.responses)})
    i = len(bad.responses) // 2
    bad.responses[i] = bytes([bad.responses[i][0] ^ 1]) + bad.responses[i][1:]
    rec, problems = Y.to_record(Y.parse_yrpx(Y.write_yrpx(bad, duel.trace)), engine, viewer=0, app="t")
    assert rec is None and len(problems) == 4  # no add way and order reproduces the recorded messages


def test_malformed_bytes_raise_value_error(replay):
    _, _, _, _, data = replay
    rng = random.Random(3)
    for n in [0, 10, 71, 72, 80, len(data) // 2]:
        with pytest.raises(ValueError):
            Y.parse_yrpx(data[:n])
    # the last byte is the LZMA end marker: without it the body is still complete and must parse identically
    assert Y.parse_yrpx(data[:-1]).yrp == Y.parse_yrpx(data).yrp
    for _ in range(50):
        b = bytearray(data)
        for _ in range(4):
            b[rng.randrange(72, len(b))] = rng.randrange(256)
        try:
            Y.parse_yrpx(bytes(b))
        except (ValueError, EOFError, __import__("lzma").LZMAError, UnicodeDecodeError):
            pass


def test_hint_order_does_not_matter(replay, engine):
    """CARD_HINT/PLAYER_HINT runs follow heap order and differ between processes (NOTEBOOK 2026-10-01): a replay
    whose hint runs are in another order is still the same duel."""
    order, duel, _, y, _ = replay
    trace = list(duel.trace)
    hints = (C.MSG_CARD_HINT, C.MSG_PLAYER_HINT)
    i = next(i for i in range(len(trace) - 1) if trace[i][0] in hints and trace[i + 1][0] in hints
             and trace[i] != trace[i + 1])
    trace[i], trace[i + 1] = trace[i + 1], trace[i]
    rec, problems = Y.to_record(Y.parse_yrpx(Y.write_yrpx(y, trace)), engine, viewer=0, app="t")
    assert problems == [] and tuple(rec.get("add_order", (0, 1))) == order


def test_real_multirole_replay_imports(engine):
    """A .yrpX made by a real Multirole (gate 4 of A2b-2, room creator seated second): Multirole adds every main-deck
    card at sequence 0 and records no select messages, and this duel only reproduces with team 1's cards added first
    (NOTEBOOK 2026-11-01 "A2b-2")."""
    from pathlib import Path

    data = (Path(__file__).parents[1] / "fixtures/yrpx/multirole-order10.yrpx").read_bytes()
    x = Y.parse_yrpx(data)
    assert not any(m[0] in Y._UNRECORDED for m in x.messages)
    rec, problems = Y.to_record(x, engine, viewer=None, app="t")
    assert problems == [] and rec is not None
    assert rec["deck_add"] == "edopro" and rec["add_order"] == [1, 0]
    assert MR.validate(rec) == [] and MR.check(rec, engine) == []
    assert rec["result"]["winner"] in (0, 1) and rec["result"]["turns"] == 35
