""" "Win this turn" puzzle unit tests (web-ux-plan step 5, item 8)."""

import json
from pathlib import Path

from ygo_zero.analysis.events import deck_path
from ygo_zero.analysis.puzzles import (
    difficulty,
    mine_candidates,
    replay_false,
    verify_puzzle,
)
from ygo_zero.decks import load_ydk
from ygo_zero.edopro.product_bot import DriverError
from ygo_zero.engine.duel import DuelConfig, Engine

FIXTURE = Path("tests/fixtures/puzzle_game.jsonl")


def _record():
    return json.loads(FIXTURE.read_text(encoding="utf-8").strip().splitlines()[0])


def _cfg(rec):
    decks = [load_ydk(deck_path(d)) for d in rec["decks"]]
    return DuelConfig(decks[0], decks[1], seed=rec["seed"])


class _FakeDriver:
    """ProductDriver-shaped stub: canned bot choices, precomputed decisions."""

    def __init__(self, decisions, bot_choices):
        self._decisions = list(decisions)
        self._bot = list(bot_choices)
        self.fed: list[int] = []

    def feed_human(self, indices):
        self.fed.extend(int(i) for i in indices)

    def decision(self, k, timeout=300):
        if k >= len(self._decisions):
            raise DriverError(f"fake product game ended before outer decision {k}")
        return dict(self._decisions[k])

    def next_bot(self, timeout=None):
        if not self._bot:
            raise DriverError("fake product game has no further bot decision")
        return int(self._bot.pop(0))

    def join(self, timeout=None):
        return None

    def close(self):
        return None


class _FakeLauncher:
    def __init__(self, decisions, bot_choices):
        self._decisions = decisions
        self._bot_choices = bot_choices

    def start_duel(self, engine, cfg, bot_seat, game_seed=0, ws_seed=0, prefix=(), prefix_decisions=None):
        assert list(prefix_decisions or []) == [d for d in self._decisions[: len(prefix_decisions or [])]]
        return _FakeDriver(self._decisions, self._bot_choices)

    def record_info(self, seed):
        return {}


def _fake_parts(engine, rec, puzzle):
    """Decisions + bot choices from a host-space replay of prefix + solution."""
    from ygo_zero.engine.env import PyEnv

    cfg = _cfg(rec)
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    decisions, bot_choices = [], []
    n_prefix = len(puzzle["prefix"])
    for i, c in enumerate(puzzle["prefix"] + puzzle["solution"]):
        if env.done:
            break
        if len(env.opts) > 1:
            decisions.append({"player": int(env.player), "n_options": len(env.opts)})
            # The driver replays the prefix verbatim internally: only live
            # (solution-phase) bot choices are ever requested via next_bot.
            if i >= n_prefix and int(env.player) != int(puzzle["seat"]):
                bot_choices.append(int(c))
        env.step(int(c))
    return decisions, bot_choices


def test_mining_finds_expected_candidate():
    engine = Engine()
    rec = _record()
    candidates, _stats = mine_candidates(engine, [rec])
    assert len(candidates) == 1
    c = candidates[0]
    assert c["id"] == "BestowedDragon2-vs-BestowedDragon-t2p1-9000"
    assert c["seat"] == rec["winner"] == 1
    assert c["turn"] == 2
    assert c["n_moves"] == 48
    assert c["difficulty"] == difficulty(48) == "hard"
    # Prefix + solution replay the full host-space game exactly.
    _, steps = replay_false(engine, _cfg(rec), [int(x) for x in rec["choices"]])
    assert [s["choice"] for s in steps] == c["prefix"] + c["solution"]
    assert [s["player"] for s in steps[len(c["prefix"]) :]] == c["actors"]


def test_verify_ok_with_replayed_bot():
    engine = Engine()
    rec = _record()
    candidates, _ = mine_candidates(engine, [rec])
    puzzle = candidates[0]
    decisions, bot_choices = _fake_parts(engine, rec, puzzle)
    launcher = _FakeLauncher(decisions, bot_choices)
    ok, reason = verify_puzzle(engine, puzzle, launcher)
    assert ok, reason


def test_verify_rejects_divergent_bot():
    engine = Engine()
    rec = _record()
    candidates, _ = mine_candidates(engine, [rec])
    puzzle = candidates[0]
    decisions, bot_choices = _fake_parts(engine, rec, puzzle)
    assert bot_choices, "fixture puzzle needs a live bot decision to diverge on"
    # The bot's first live answer differs (still a legal index).
    k0 = next(i for i, d in enumerate(decisions) if d["player"] != puzzle["seat"])
    bad = list(bot_choices)
    bad[0] = (bad[0] + 1) % decisions[k0]["n_options"]
    assert bad[0] != bot_choices[0]
    launcher = _FakeLauncher(decisions, bad)
    ok, reason = verify_puzzle(engine, puzzle, launcher)
    assert not ok
    assert reason, "a rejection always carries a reason"
