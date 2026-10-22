"""EDOPro replay export (demos/yrp.py): the file parses back and replays to the same game the way EDOPro sets it up."""

import random

from decklight.demos import yrp as Y
from decklight.engine.duel import DuelConfig
from decklight.engine.env import PyEnv


def _game(engine, decks, seed=3):
    cfg = DuelConfig(decks["Branded"], decks["Labrynth"], seed=seed, lp=1000)  # low LP: short game
    env, rng = PyEnv(engine, cfg), random.Random(seed)
    obs = env.reset()
    for _ in range(3000):
        if obs is None:
            break
        obs = env.step(rng.randrange(int(obs["action_mask"].sum())))
    assert env.done
    return cfg, env


def test_roundtrip_and_verify(engine, decks):
    cfg, env = _game(engine, decks)
    data = Y.to_yrp(cfg, env.duel.responses, ("Our AI", "ygo-agent"))
    y = Y.parse_yrp(data)
    assert y.names == ["Our AI", "ygo-agent"] and y.lp == 1000 and y.flags == cfg.flags
    assert y.responses == env.duel.responses
    assert y.decks[0][0] == Y.edopro_main(cfg.main_of(0)) and y.decks[1][1] == list(cfg.deck1.extra)
    assert Y.verify(engine, data, env.duel) is None


def test_verify_catches_a_cut_game(engine, decks):
    cfg, env = _game(engine, decks)
    data = Y.to_yrp(cfg, env.duel.responses[:-1], ("a", "b"))
    assert Y.verify(engine, data, env.duel) is not None
