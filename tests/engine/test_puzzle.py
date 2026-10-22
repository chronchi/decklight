"""Puzzle start: cards placed before the duel starts (DuelConfig.setup)."""

import random

import numpy as np
import pytest

from decklight.decks import Deck
from decklight.engine import constants as C
from decklight.engine import obs as O
from decklight.engine.duel import DuelConfig, Placement
from decklight.engine.env import PyEnv


def _puzzle(decks, seed=1):
    """Player 0 (Branded): a chosen hand of 3 + a monster in zone 2. Player 1: 1 card in hand, 1 in GY. No draws."""
    branded = decks["Branded"]
    hand = [25451383, 32731036, 68468459]  # three main-deck cards
    main = list(branded.main)
    for code in hand:
        main.remove(code)
    setup = tuple(Placement(0, c, C.LOCATION_HAND) for c in hand) + (
        Placement(0, 62962630, C.LOCATION_MZONE, 2),
        Placement(1, 14558127, C.LOCATION_HAND),
        Placement(1, 14558127, C.LOCATION_GRAVE),
    )
    main.remove(62962630)
    deck0 = Deck(tuple(main), branded.extra, ())
    return DuelConfig(deck0, decks["Labrynth"], seed=seed, hand=0, setup=setup)


def test_puzzle_board_is_as_placed(engine, decks):
    env = PyEnv(engine, _puzzle(decks))
    obs = env.reset()
    assert env.player == 0 and env.duel.turn == 1
    cards = obs["cards"][obs["card_mask"]]
    loc, ctrl = cards[:, 1], cards[:, 3]
    mine_hand = cards[(loc == O.LOC_ID[C.LOCATION_HAND]) & (ctrl == 0)]
    # going first: no draw on turn 1, so the hand is exactly what was placed
    assert sorted(mine_hand[:, 0]) == sorted([25451383, 32731036, 68468459])
    mz = cards[(loc == O.LOC_ID[C.LOCATION_MZONE]) & (ctrl == 0)]
    assert mz[:, 0].tolist() == [62962630] and mz[0, 2] == 3 and mz[0, 5] == 1  # zone 2, FU-ATK
    opp = cards[ctrl == 1]
    assert sorted(opp[:, 1].tolist()) == [O.LOC_ID[C.LOCATION_HAND], O.LOC_ID[C.LOCATION_GRAVE]]
    assert opp[opp[:, 1] == O.LOC_ID[C.LOCATION_HAND], 0].tolist() == [0]  # hidden
    assert opp[opp[:, 1] == O.LOC_ID[C.LOCATION_GRAVE], 0].tolist() == [14558127]
    g = obs["global"]
    assert g[6] == len(decks["Branded"].main) - 4 and g[7] == 3 and g[8] == 1


def test_puzzle_plays_out_and_matches_cpp(engine, decks):
    cpp_env = pytest.importorskip("decklight.engine.cpp_env")
    config = _puzzle(decks, seed=3)
    py, cc = PyEnv(engine, config), cpp_env.CppEnv(engine, config)
    a, b, rng = py.reset(), cc.reset(), random.Random(0)
    while a is not None:
        assert all(np.array_equal(a[k], b[k]) for k in a)
        i = rng.randrange(int(a["action_mask"].sum()))
        a, b = py.step(i), cc.step(i)
    assert b is None and py.winner == cc.winner
