import random

import numpy as np
import pytest

from decklight.engine import constants as C
from decklight.engine import obs as O
from decklight.engine.duel import DuelConfig
from decklight.engine.env import PyEnv


def _play(env, seed, max_steps=3000):
    rng, obs, trail = random.Random(seed), env.reset(), []
    while obs is not None and len(trail) < max_steps:
        trail.append(obs)
        obs = env.step(rng.randrange(int(obs["action_mask"].sum())))
    return trail


@pytest.mark.parametrize("seed", range(4))
def test_random_env_games_finish(engine, decks, seed):
    names = sorted(decks)
    env = PyEnv(engine, DuelConfig(decks[names[seed]], decks[names[(seed + 1) % 4]], seed=seed))
    trail = _play(env, seed)
    assert env.done and env.winner in (0, 1, 2)
    for obs in trail:
        n = int(obs["action_mask"].sum())
        assert n >= 2  # single-option steps are automatic
        assert obs["global"][23] == O.OBS_VERSION
        rows = obs["actions"][:n, 1]
        assert ((rows >= -1) & (rows < obs["card_mask"].sum())).all()


def test_replay_reproduces_observations(engine, decks):
    config = DuelConfig(decks["Branded"], decks["Labrynth"], seed=5)
    env = PyEnv(engine, config)
    trail = _play(env, 5, max_steps=400)
    again = PyEnv.replay(engine, config, env.choices[:200])
    assert all(np.array_equal(again_obs, trail[200][k]) for k, again_obs in again._observe().items())


def test_hidden_information(engine, decks):
    env = PyEnv(engine, DuelConfig(decks["SkyStrikerAce"], decks["BlueEyes"], seed=1))
    obs = env.reset()
    me = env.player
    cards = obs["cards"][obs["card_mask"]]
    loc, ctrl = cards[:, 1], cards[:, 3]
    opp_hand = cards[(loc == O.LOC_ID[C.LOCATION_HAND]) & (ctrl == 1)]
    assert len(opp_hand) > 0 and (opp_hand[:, 0] == 0).all() and (opp_hand[:, 7:] == 0).all()
    assert not ((loc == O.LOC_ID[C.LOCATION_DECK]) & (ctrl == 1)).any()  # no opp deck rows
    my_deck = cards[(loc == O.LOC_ID[C.LOCATION_DECK]) & (ctrl == 0)]
    assert (my_deck[:, 2] == 0).all() and (np.diff(my_deck[:, 0]) >= 0).all()  # unordered multiset
    g = obs["global"]
    assert g[6] == len(my_deck) and g[0] == g[1] == 8000
    assert g[5] == int(env.duel.turn_player == me)
