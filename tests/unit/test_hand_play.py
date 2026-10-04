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
