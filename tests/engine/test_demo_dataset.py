"""Verified demo records replay exactly and turn into BC samples."""

from collections import Counter
from pathlib import Path

import pytest

from ygo_zero.demos.dataset import iter_samples, load_records, record_config
from ygo_zero.engine.cpp_env import CppEnv

VERIFIED = Path(__file__).resolve().parents[2] / "data" / "demos" / "verified"


def _records():
    if not VERIFIED.exists():
        pytest.skip("no verified demos (data/ is local)")
    recs = load_records(VERIFIED)
    if not recs:
        pytest.skip("no verified demos")
    return recs


def test_every_verified_record_replays_to_its_end_board(engine):
    for rec in _records():
        env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
        obs = env.reset()
        for c in rec["choices"]:
            obs = env.step(c)
        assert obs is not None, rec["line_id"]  # a turn-1 line never ends the duel
        rows = obs["cards"][obs["card_mask"]]
        mine = rows[rows[:, 3] == env.player]  # controller is relative to the player to act; I am player 0
        field = Counter(mine[(mine[:, 1] == 3) | (mine[:, 1] == 4)][:, 0].tolist())
        need = Counter(c for c in rec["end_board_codes"] if c is not None)
        assert all(field[c] >= k for c, k in need.items()), rec["line_id"]


def test_samples_have_legal_actions_and_weights(engine):
    recs = _records()
    n = 0
    for obs, action, weight, meta in iter_samples(engine, VERIFIED):
        assert 0 <= action < int(obs["action_mask"].sum())
        assert weight in (1.0, 0.5) and meta["label"] in ("stated", "inferred")
        n += 1
    assert n >= len(recs)
