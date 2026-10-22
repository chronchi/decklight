"""Quality B: strategy notes in game prompts + going-second openings. No network, LLM or engine search."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from decklight.decks import Deck
from decklight.demos import game as G
from decklight.engine import constants as C
from decklight.engine.actions import Kind
from decklight.engine.duel import DuelConfig

ROOT = Path(__file__).resolve().parents[2]

_HINT = (
    "Plan how to break or play through the opponent's board (board breakers, removal, "
    "hand traps you already used), then build or attack for lethal if possible."
)


def _state(went: str, opp_mons: str = "none", opp_st: str = "none") -> str:
    return "\n".join([
        f"Turn 2 (your turn), phase Main 1. LP: you 8000, opponent 8000. You went {went}.",
        "Your hand: Ash Blossom & Joyous Spring",
        "Your monsters: none",
        "Your spells/traps: none",
        "Your GY: none",
        "Your banished: none",
        "Your Extra Deck: none",
        "Your Deck: 35 cards",
        f"Opponent monsters: {opp_mons}",
        f"Opponent spells/traps: {opp_st}",
        "Opponent GY: none",
        "Opponent banished: none",
        "Opponent hand: 5 cards; Deck: 35 cards",
    ])


# --- strategy notes in prompts ---


def test_turn_prompt_notes_present_and_absent(monkeypatch):
    monkeypatch.setattr(G, "notes_block", lambda d: f"\nSTRATEGY NOTES for {d}:\nDO THE THING\n")
    p = G.turn_prompt("mikanko", "SHEET", _state("first"), False, [])
    assert "STRATEGY NOTES for mikanko" in p and "DO THE THING" in p
    monkeypatch.setattr(G, "notes_block", lambda d: "")
    assert "STRATEGY" not in G.turn_prompt("mikanko", "SHEET", _state("first"), False, [])


def test_interrupt_prompt_notes_present_and_absent(monkeypatch):
    monkeypatch.setattr(G, "notes_block", lambda d: f"\nSTRATEGY NOTES for {d}:\nWAIT FOR IT\n")
    p = G.interrupt_prompt("sky-striker", _state("first"), [], [], "0: PASS (no card)")
    assert "STRATEGY NOTES for sky-striker" in p and "WAIT FOR IT" in p
    monkeypatch.setattr(G, "notes_block", lambda d: "")
    assert "STRATEGY" not in G.interrupt_prompt("sky-striker", _state("first"), [], [], "0: PASS (no card)")


def test_turn_prompt_going_second_hint(monkeypatch):
    monkeypatch.setattr(G, "notes_block", lambda d: "")
    board = "Blue-Eyes White Dragon [face-up ATK, 3000/2500, zone 0]"
    assert _HINT in G.turn_prompt("d", "S", _state("second", opp_mons=board), True, [])
    assert _HINT in G.turn_prompt("d", "S", _state("second", opp_st="Infinite Impermanence [set]"), True, [])
    assert _HINT not in G.turn_prompt("d", "S", _state("second"), True, [])  # empty board
    assert _HINT not in G.turn_prompt("d", "S", _state("first", opp_mons=board), False, [])


# --- opening replay (fake env, no engine) ---


def _obs(turn: int, my_turn: bool, idle: bool) -> dict:
    g = np.zeros(24, dtype=np.int32)
    g[2], g[3], g[4], g[5] = turn, 0x04, 0, 1 if my_turn else 0
    g[20], g[21] = 0, C.MSG_SELECT_IDLECMD if idle else C.MSG_SELECT_YESNO
    acts = np.zeros((64, 10), dtype=np.int32)
    amask = np.zeros(64, bool)
    acts[0, 0], acts[0, 2] = int(Kind.TO_END), 0
    amask[0] = True
    return {
        "global": g, "cards": np.zeros((160, 21), dtype=np.int32), "card_mask": np.zeros(160, bool),
        "chain": np.zeros((16, 5), dtype=np.int32), "chain_mask": np.zeros(16, bool),
        "actions": acts, "action_mask": amask,
    }


class _EnvContinue:
    """Turn-1 prefix of 2 choices, then an open turn-2 state for player 1, then the duel ends."""

    def __init__(self, engine, config):
        self.config = config
        self.player = 0
        self.steps = 0
        self.winner = None

    def reset(self):
        return _obs(turn=1, my_turn=True, idle=False)

    def step(self, i):
        self.steps += 1
        if self.steps < 2:
            return _obs(turn=1, my_turn=True, idle=False)
        if self.steps == 2:
            self.player = 1
            return _obs(turn=2, my_turn=True, idle=True)
        self.winner = 1
        return None


class _EnvEarlyEnd:
    """The duel ends after the first choice (an opening prefix longer than that is invalid)."""

    def __init__(self, engine, config):
        self.config = config
        self.player = 0
        self.steps = 0
        self.winner = 0

    def reset(self):
        return _obs(turn=1, my_turn=True, idle=False)

    def step(self, i):
        self.steps += 1
        # implicit None: the duel ends right after the first choice


def _decks():
    return (("a", Deck((89631139,), (), (), "a")), ("b", Deck((8240199,), (), (), "b")))


def _play(monkeypatch, env_cls, opening, **kw):
    monkeypatch.setattr(G, "CppEnv", env_cls)
    monkeypatch.setattr(G, "card_sheet", lambda e, d, n: "SHEET")
    calls = []

    def comp(prompt):
        calls.append(prompt)
        return "not json"

    decks = _decks()
    rec = G.play_game(
        object(), decks, 5, comp, SimpleNamespace(names={}), repairs=0, interrupts=0,
        plan_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not verify")),
        opening=opening, log=lambda s: None, **kw,
    )
    return rec, calls


def _opening(choices=(0, 0)):
    decks = _decks()
    cfg = DuelConfig(decks[0][1], decks[1][1], seed=7)
    return (cfg, list(choices), ["stated", "inferred"][: len(choices)])


def test_opening_replay_then_turn_two(monkeypatch):
    rec, calls = _play(monkeypatch, _EnvContinue, _opening())
    assert rec["choices"] == [0, 0, 0] and rec["labels"] == ["stated", "inferred", "fallback"]
    assert rec["actors"] == [0, 0, 1] and rec["turns"] == [1, 1, 2]
    assert rec["opening"] is None  # the driver fills in the source record's line_id
    assert len(calls) == 1  # exactly one LLM plan: turn 2 against the established board
    assert rec["winner"] == 1


# Openings are no longer replayed blindly (2026-10-03): the line is player 0's turn-1 plan, so the second player can
# interrupt it. Bad openings are rejected up front by `build_opening`/`opening_valid` (tested below).


def test_opening_valid(monkeypatch):
    monkeypatch.setattr(G, "CppEnv", _EnvContinue)
    cfg = _opening()[0]
    assert G.opening_valid(object(), cfg, [0, 0])
    assert not G.opening_valid(object(), cfg, [0, 9])
    monkeypatch.setattr(G, "CppEnv", _EnvEarlyEnd)
    assert not G.opening_valid(object(), cfg, [0, 0])


# --- deck1 replacement + candidate ranking ---


def _rec(lid: str, n_end: int, n_choices: int = 4, seed: int = 0, labels=None) -> dict:
    return {
        "line_id": lid,
        "config": {
            "deck0_main": [89631139], "deck0_extra": [], "deck1_main": [8240199], "deck1_extra": [],
            "seed": seed, "hand": 0, "draw": 1, "setup": [],
        },
        "choices": [0] * n_choices,
        "labels": labels if labels is not None else ["stated"] * n_choices,
        "end_board": [f"card{i}" for i in range(n_end)],
    }


def test_rank_openings_prefers_biggest_boards():
    recs = [_rec("a/small", 1), _rec("a/big", 3), _rec("b/huge", 9), _rec("a/mid", 2)]
    ranked = G.rank_openings(recs, "a")
    assert [r["line_id"] for r in ranked] == ["a/big", "a/mid", "a/small"]


def test_rank_openings_dedupes_and_filters():
    recs = [_rec("a/x", 2), _rec("a/x", 2), _rec("a/bad", 5, labels=["stated"])]
    assert [r["line_id"] for r in G.rank_openings(recs, "a")] == ["a/x"]


def test_find_openings_searches_listed_layouts(tmp_path):
    import json

    (tmp_path / "verified").mkdir()
    (tmp_path / "verified" / "a.jsonl").write_text(json.dumps(_rec("a/v", 1)) + "\n")
    (tmp_path / "sub" / "verified").mkdir(parents=True)
    (tmp_path / "sub" / "verified" / "a.jsonl").write_text(json.dumps(_rec("a/s", 2)) + "\n")
    (tmp_path / "planned-q" / "verified").mkdir(parents=True)
    (tmp_path / "planned-q" / "verified" / "a.jsonl").write_text(json.dumps(_rec("a/p", 3)) + "\n")
    imp = tmp_path / "planned-q" / "improved" / "verified"
    imp.mkdir(parents=True)
    (imp / "a.jsonl").write_text(json.dumps(_rec("a/i", 4)) + "\n")
    got = G.find_openings("a", tmp_path)
    assert [r["line_id"] for r in got] == ["a/i", "a/p", "a/s", "a/v"]


def test_build_opening_replaces_deck1(monkeypatch):
    monkeypatch.setattr(G, "CppEnv", _EnvContinue)
    deck_b = Deck(tuple(range(1, 8)), (), (), "b")
    out = G.build_opening(object(), _rec("a/line", 2, n_choices=2), deck_b, seed=3)
    assert out is not None
    cfg, choices, labels = out
    hand = [pl.code for pl in cfg.setup if pl.team == 1]
    assert len(hand) == G.OPENING_HAND and len(cfg.deck1.main) == 2  # opponent gets a real opening hand
    assert sorted(hand + list(cfg.deck1.main)) == list(deck_b.main) and list(cfg.deck0.main) == [89631139]
    assert choices == [0, 0] and labels == ["stated"] * 2


def test_build_opening_skips_invalid(monkeypatch):
    monkeypatch.setattr(G, "CppEnv", _EnvEarlyEnd)
    assert G.build_opening(object(), _rec("a/line", 2, n_choices=2), Deck((1,), (), (), "b")) is None
    monkeypatch.setattr(G, "CppEnv", _EnvContinue)
    assert G.build_opening(object(), _rec("a/bad", 2, labels=["stated"]), Deck((1,), (), (), "b")) is None


def test_play_games_help_shows_new_flags():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "play_games.py"), "--help"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert r.returncode == 0
    for flag in ("--from-line", "--from-index", "--random-records", "--opening-frac"):
        assert flag in r.stdout
