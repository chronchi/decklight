import pytest

from ygo_zero.engine.duel import Duel, DuelConfig, canonical_trace
from ygo_zero.engine.random_agent import RandomAgent


def _play(engine, config, seed):
    duel = engine.new_duel(config)
    winner = duel.run([RandomAgent(seed, engine.cards), RandomAgent(seed + 1, engine.cards)])
    return duel, winner


@pytest.mark.parametrize("seed", range(6))
def test_random_duel_finishes(engine, decks, seed):
    names = sorted(decks)
    config = DuelConfig(decks[names[seed % 4]], decks[names[(seed + 1) % 4]], seed=seed)
    duel, winner = _play(engine, config, seed)
    assert winner in (0, 1, 2), "hit max_steps"
    assert duel.turn >= 1
    assert not [line for t, line in duel.log if t == 0], "script errors"


def test_same_seed_same_game(engine, decks):
    config = DuelConfig(decks["Branded"], decks["BlueEyes"], seed=42)
    a, wa = _play(engine, config, 7)
    b, wb = _play(engine, config, 7)
    assert wa == wb and a.responses == b.responses
    assert canonical_trace(a.trace) == canonical_trace(b.trace)


def test_replay_rebuilds_game_and_clones_midgame(engine, decks):
    config = DuelConfig(decks["Labrynth"], decks["SkyStrikerAce"], seed=3)
    full, winner = _play(engine, config, 11)

    replayed = Duel.replay(engine, config, full.responses)
    assert replayed.ended and replayed.winner == winner
    assert canonical_trace(replayed.trace) == canonical_trace(full.trace)

    # Clone at the midpoint and continue with the original responses: same game.
    k = len(full.responses) // 2
    clone = Duel.replay(engine, config, full.responses[:k])
    assert clone.pending is not None
    for r in full.responses[k:]:
        clone.respond(r)
        clone.advance()
    assert clone.winner == winner
    assert canonical_trace(clone.trace) == canonical_trace(full.trace)
