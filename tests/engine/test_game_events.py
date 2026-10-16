"""Unit tests for the game event extractor (engine replay only, no model)."""

import json
import random

import pytest

from ygo_zero.analysis.events import COUNTERS, events
from ygo_zero.engine.duel import DuelConfig
from ygo_zero.engine.env import PyEnv


def _play_random(engine, cfg: DuelConfig, seed: int) -> PyEnv:
    rng = random.Random(seed)
    env = PyEnv(engine, cfg)
    obs = env.reset()
    while obs is not None:
        obs = env.step(rng.randrange(int(obs["action_mask"].sum())))
    return env


def _rec_of(env: PyEnv, game_idx: int, seed: int, winner) -> dict:
    return {
        "game_idx": game_idx,
        # Deck-like objects pass straight through; the CLI passes ydk names instead.
        "decks": [env.duel.config.deck0, env.duel.config.deck1],
        "seed": seed,
        "choices": list(env.choices),
        "winner": winner,
    }


def test_events_random_game(engine, decks):
    names = sorted(decks)
    cfg = DuelConfig(decks[names[1 % len(names)]], decks[names[2 % len(names)]], seed=1)
    env = _play_random(engine, cfg, 1)
    assert env.done and env.winner in (0, 1)
    rec = _rec_of(env, 0, 1, env.winner)
    out = events(rec, engine)
    # Winner invariant.
    assert out["winner"] == env.winner == rec["winner"]
    # Turns counted from 1, contiguous, with a turn player.
    turns = out["turns"]
    assert [t["turn"] for t in turns] == list(range(1, len(turns) + 1))
    assert all(t["turn_player"] in (0, 1) for t in turns)
    # Every per-player counter present.
    for t in turns:
        assert len(t["players"]) == 2
        for p in t["players"]:
            assert set(p) == set(COUNTERS)
            assert all(v >= 0 for v in p.values())
    # Opp-turn activations are a subset of activations.
    for t in turns:
        for p in t["players"]:
            assert p["activations_opp_turn"] <= p["activations"]
    # Every chain link has a controller; negated links were suffered by their controller.
    for link in out["chain_links"]:
        assert link["controller"] in (0, 1)
        assert link["chain_len_at_activation"] >= 1
    n_neg_links = sum(1 for link in out["chain_links"] if link["negated"])
    n_neg_suffered = sum(p["negated_by_p_suffered"] for t in turns for p in t["players"])
    assert n_neg_links == n_neg_suffered
    # LP formula: damage + paid - recovered == 8000 - final LP per seat.
    for p in (0, 1):
        dmg = sum(t["players"][p]["damage_taken"] for t in turns)
        paid = sum(t["players"][p]["lp_paid"] for t in turns)
        got = sum(t["players"][p]["lp_recovered"] for t in turns)
        assert dmg + paid - got == 8000 - out["final_lp"][p]
    # Cards seen are sorted code lists; activations imply exposure.
    assert len(out["cards_seen"]) == 2
    for seen in out["cards_seen"]:
        assert seen == sorted(seen) and all(c > 0 for c in seen)
    # JSON-serializable (the CLI writes one line per game).
    json.dumps(out)


def test_events_winner_mismatch_raises(engine, decks):
    names = sorted(decks)
    cfg = DuelConfig(decks[names[0]], decks[names[1]], seed=2)
    env = _play_random(engine, cfg, 2)
    assert env.done
    rec = _rec_of(env, 0, 2, 1 - env.winner if env.winner in (0, 1) else 0)
    with pytest.raises(ValueError, match="replay winner"):
        events(rec, engine)
