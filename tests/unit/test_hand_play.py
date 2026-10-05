"""Hand-play tooling: option descriptions, token matching, PASS* seat stop."""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


hand_play = _load("hand_play")


def test_desc_text_system_string():
    from ygo_zero.demos.game import desc_text

    assert desc_text(0, 573) == "Add the card(s) to your hand"


def test_desc_text_card_str():
    from ygo_zero.demos.game import desc_text
    from ygo_zero.engine.resources import default_cdbs

    code, want = None, None
    for path in default_cdbs():
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT id, str1 FROM texts WHERE str1 IS NOT NULL AND str1 != '' LIMIT 1"
            ).fetchone()
        finally:
            con.close()
        if row is not None:
            code, want = int(row[0]), str(row[1]).strip()
            break
    assert code is not None, "no card with str1 in the cdb"
    assert desc_text(code, 0) == want


def _fake_obs(kinds_codes):
    from ygo_zero.engine.actions import Kind

    assert Kind.SELECT is not None
    actions = [[int(k), -1, code, 0, 0, 0, 0, 0, 0] for k, code in kinds_codes]
    return {"actions": actions, "action_mask": [1] * len(actions)}


def test_match_exact_name_beats_substring():
    from ygo_zero.engine.actions import Kind

    c1, c2 = 1111, 2222
    obs = _fake_obs([(Kind.SELECT, c1), (Kind.SELECT, c2)])
    names = {c1: "Farewelcome Labrynth", c2: "Welcome Labrynth"}
    assert hand_play._match(obs, "SEL:Welcome_Labrynth", names) == 1


def test_pass_star_stops_at_undeclined_seat(tmp_path, monkeypatch):
    from ygo_zero.engine.duel import Engine

    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    st = hand_play._load("g")
    assert st["decks"] == ["snake-eye", "tenpai-dragon"] and st["seed"] == 11 and st["choices"] == []

    # a dumb option-0 policy to a P0 response window on the opponent's turn (deterministic, seed 11)
    engine = Engine()
    env, obs = hand_play._replay(engine, st)
    for _ in range(500):
        assert obs is not None, "duel ended before any P0 response window"
        msg, my_turn = int(obs["global"][21]), bool(int(obs["global"][5]))
        if env.player == 0 and not my_turn and msg == hand_play.C.MSG_SELECT_CHAIN:
            break
        st["choices"].append(0)
        st["labels"].append("x")
        st["actors"].append(env.player)
        st["turns"].append(int(obs["global"][2]))
        obs = env.step(0)
        obs = hand_play._auto(env, obs, st)
    else:
        raise AssertionError("no P0 response window on the opponent's turn within 500 decisions")

    n0 = len(st["choices"])
    obs, reason = hand_play._pass_star(env, obs, st)
    n_pass = sum(1 for lab in st["labels"][n0:] if lab == "pass")
    assert 1 <= n_pass <= hand_play._PASS_STEPS
    seats = {act for lab, act in zip(st["labels"][n0:], st["actors"][n0:], strict=True) if lab == "pass"}
    # P0 issued PASS*; P1 has not declined its options since the last action, so its window is a real decision
    assert seats == {0}, (seats, reason)
    assert reason.startswith("P1 has something to decide"), reason
    # P1 now passes too: P0's window in the same turn, with no action since its decline, is passed automatically
    n1 = len(st["choices"])
    obs, reason = hand_play._pass_star(env, obs, st)
    assert not reason.startswith("P0 has something"), reason
    assert st["actors"][n1] == 1
    state = json.loads((tmp_path / "work" / "g" / "state.json").read_text())
    assert state["choices"] == []


def test_numeric_token_stops_on_prompt_change(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    capsys.readouterr()
    # the second "0" refers to the first prompt's numbering, so it must stop, not spill over
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "do", "g", "0 0"])
    hand_play.main()
    out = capsys.readouterr().out
    assert "prompt changed" in out
    st = hand_play._load("g")
    assert st["labels"] == ["gold", "auto"], st["labels"]  # only the first token applied


def _fake_obs_loc(items):
    """Fake obs with full 10-col option rows: items = (kind, code, loc_id, rel_ctrl)."""
    actions = [[int(k), -1, code, 0, 0, 0, 0, loc, ctrl, 0] for k, code, loc, ctrl in items]
    return {"actions": actions, "action_mask": [1] * len(actions)}


def test_duplicate_names_get_location_suffix_and_match():
    from ygo_zero.engine.actions import Kind

    code = 99999
    obs = _fake_obs_loc([(Kind.SELECT, code, 5, 0), (Kind.SELECT, code, 2, 0)])
    names = {code: "Snake-Eye Ash"}
    disp = hand_play._display_names(obs, names)
    assert disp == ["Snake-Eye Ash (GY)", "Snake-Eye Ash (hand)"], disp
    assert hand_play._match(obs, "SEL:Snake-Eye_Ash_(GY)", names) == 0
    assert hand_play._match(obs, "SEL:Snake-Eye_Ash_(hand)", names) == 1
    # opponent's copy is labeled and matchable too
    obs2 = _fake_obs_loc([(Kind.SELECT, code, 5, 0), (Kind.SELECT, code, 5, 1)])
    disp2 = hand_play._display_names(obs2, names)
    assert disp2[1] == "Snake-Eye Ash (opponent's GY)", disp2
    assert hand_play._match(obs2, "SEL:Snake-Eye_Ash_(GY)", names) == 0


def test_check_runs_on_fresh_duel(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "check", "g"])
    hand_play.main()
    out = capsys.readouterr().out
    assert "Unused options:" in out
    assert "Already activated this turn" in out
    assert "Opponent LP" in out
    st = hand_play._load("g")  # check changes no state
    assert st["choices"] == []


def test_compact_view_hides_unchanged_extra_deck(tmp_path, monkeypatch, capsys):
    from ygo_zero.demos.names import CardNames
    from ygo_zero.engine.duel import Engine

    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    st = hand_play._load("g")
    env, obs = hand_play._replay(Engine(), st)
    names = CardNames()
    capsys.readouterr()
    hand_play._show(env, obs, st, names, False)  # first view of the seat: full
    full_out = capsys.readouterr().out
    hand_play._show(env, obs, st, names, False, compact=True)  # same seat, same turn: compact
    comp_out = capsys.readouterr().out
    full_ed = next(line for line in full_out.splitlines() if line.startswith("Your Extra Deck:"))
    comp_ed = next(line for line in comp_out.splitlines() if line.startswith("Your Extra Deck:"))
    assert full_ed != comp_ed and comp_ed.endswith("cards")  # count only, e.g. "15 cards"
    assert "Your GY:" not in comp_out  # unchanged since the previous view
    assert "Your hand:" in comp_out and "Your monsters:" in comp_out  # always shown


def _chain_obs(live_codes, chain_len=0, turn=3):
    """Fake MSG_SELECT_CHAIN window: ACTIVATE options for `live_codes` plus the passive option."""
    from ygo_zero.engine.actions import Kind

    acts = [[int(Kind.ACTIVATE), -1, code, 0, 0, 0, 0x10, 2, 0, 0] for code in live_codes]
    acts.append([int(Kind.PASS), -1, 0, 0, 0, 0, 0x10, 0, -1, 0])
    g = [0] * 24
    g[2], g[21] = turn, 0x10
    return {
        "actions": acts, "action_mask": [1] * len(acts),
        "global": g, "chain_mask": [1] * chain_len,
        "cards": [], "card_mask": [],
    }


class _ScriptEnv:
    """Fake env replaying a fixed obs script (same seat throughout)."""

    def __init__(self, states):
        self.states = list(states)
        self.player = 0

    def step(self, i):
        return self.states.pop(0)


def _hold_names(codes=()):
    from types import SimpleNamespace

    return SimpleNamespace(names={c: n for c, n in codes})


KIRIN, ASH = 2526224, 14558127


def _hold_st(held=()):
    return {
        "choices": [], "labels": [], "actors": [], "turns": [],
        "hold": {"0": {"turn": 3, "names": list(held)}},
    }


def test_hold_passes_empty_chain_only_held_options():
    hand_play._DECLINED.clear()
    names = _hold_names([(KIRIN, "Fire King High Avatar Kirin"), (ASH, "Ash Blossom & Joyous Spring")])
    st = _hold_st(["fire king high avatar kirin"])
    # first window declines Ash; next window offers only the held Kirin on an empty chain; then a link
    # is added to the chain and PASS* must stop there
    env = _ScriptEnv([_chain_obs([KIRIN]), _chain_obs([], chain_len=1)])
    _obs, reason = hand_play._pass_star(env, _chain_obs([ASH]), st, names)
    assert st["labels"] == ["pass", "pass"]
    assert st["actors"] == [0, 0]
    assert reason.startswith("chain changed"), reason


def test_hold_stops_at_unheld_option_and_expires_on_turn_change():
    hand_play._DECLINED.clear()
    names = _hold_names([(KIRIN, "Fire King High Avatar Kirin"), (ASH, "Ash Blossom & Joyous Spring")])
    st = _hold_st(["fire king high avatar kirin"])
    env = _ScriptEnv([_chain_obs([ASH])])
    _, reason = hand_play._pass_star(env, _chain_obs([KIRIN]), st, names)
    assert reason.startswith("P0 has something to decide"), reason
    assert st["labels"] == ["pass"]  # only the window PASS* was issued at
    assert hand_play._held(st, 0, 3) == {"fire king high avatar kirin"}
    assert hand_play._held(st, 0, 4) == set()  # holds expire on turn change
    assert hand_play._hold_covers(_chain_obs([KIRIN]), {"fire king high avatar kirin"}, names.names)
    assert not hand_play._hold_covers(_chain_obs([ASH]), {"fire king high avatar kirin"}, names.names)


def test_check_battle_section_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "check", "g"])
    hand_play.main()
    out = capsys.readouterr().out
    assert "Battle:" in out
    assert "Direct-damage total if the opponent's field were empty:" in out


def test_same_zone_duplicates_get_zone_suffix_and_match():
    import numpy as np

    from ygo_zero.engine.actions import Kind

    code = 99999
    cards = np.zeros((2, 21), dtype=np.int32)
    cards[0, 0], cards[0, 1], cards[0, 2] = code, 3, 1
    cards[1, 0], cards[1, 1], cards[1, 2] = code, 3, 3
    obs = {
        "actions": [
            [int(Kind.SELECT), 0, code, 0, 0, 0, 0, 3, 0, 0],
            [int(Kind.SELECT), 1, code, 0, 0, 0, 0, 3, 0, 0],
        ],
        "action_mask": [1, 1],
        "cards": cards, "card_mask": np.array([True, True]),
    }
    names = {code: "Snake-Eye Ash"}
    disp = hand_play._display_names(obs, names)
    assert disp == ["Snake-Eye Ash (Monster Zone 0)", "Snake-Eye Ash (Monster Zone 2)"], disp
    assert hand_play._match(obs, "SEL:Snake-Eye_Ash_(Monster_Zone_2)", names) == 1


def test_number_options_show_values_and_match():
    from ygo_zero.engine.actions import Kind

    obs = {"actions": [[int(Kind.NUMBER), -1, 0, 0, 0, 5, 143, 0, -1, 0]], "action_mask": [1]}
    assert hand_play._display_names(obs, {}) == ["5"]
    assert hand_play._match(obs, "NUM:5", {}) == 0


def test_saved_gold_games_replay_unchanged():
    """Every saved gold record replays to the end with all choices legal (hand_play logic unchanged)."""
    import json

    from ygo_zero.demos.dataset import record_config
    from ygo_zero.demos.game import _opts
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.engine.duel import Engine

    engine = Engine()
    paths = sorted((ROOT / "data" / "demos" / "gold").glob("*.jsonl"))
    assert paths, "no saved gold games"
    n_choices = 0
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
            obs = env.reset()
            for k, ch in enumerate(rec["choices"]):
                assert obs is not None, f"{rec['game_id']}: duel ended at choice {k}"
                assert 0 <= int(ch) < len(_opts(obs)), f"{rec['game_id']}: illegal choice {ch} at {k}"
                obs = env.step(int(ch))
            n_choices += len(rec["choices"])
    assert n_choices > 0

