"""Hand-play tooling: option descriptions, token matching, PASS* seat stop."""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

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
    from decklight.demos.game import desc_text

    assert desc_text(0, 573) == "Add the card(s) to your hand"


def test_desc_text_card_str():
    from decklight.demos.game import desc_text
    from decklight.engine.resources import default_cdbs

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
    from decklight.engine.actions import Kind

    assert Kind.SELECT is not None
    actions = [[int(k), -1, code, 0, 0, 0, 0, 0, 0] for k, code in kinds_codes]
    return {"actions": actions, "action_mask": [1] * len(actions)}


def test_match_exact_name_beats_substring():
    from decklight.engine.actions import Kind

    c1, c2 = 1111, 2222
    obs = _fake_obs([(Kind.SELECT, c1), (Kind.SELECT, c2)])
    names = {c1: "Farewelcome Labrynth", c2: "Welcome Labrynth"}
    assert hand_play._match(obs, "SEL:Welcome_Labrynth", names) == 1


def test_pass_star_stops_at_undeclined_seat(tmp_path, monkeypatch):
    from decklight.engine.duel import Engine

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
    from decklight.engine.actions import Kind

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
    # same name on both sides is ambiguous: the token stops, the qualifier picks
    assert hand_play._match(obs2, "SEL:Snake-Eye_Ash_(GY)", names) is None
    assert hand_play._match(obs2, "SEL:own:Snake-Eye_Ash_(GY)", names) == 0
    assert hand_play._match(obs2, "SEL:opp:Snake-Eye_Ash_(GY)", names) == 1
    msg = hand_play._no_match_msg(obs2, "SEL:Snake-Eye_Ash_(GY)", names)
    assert "matches several options" in msg and "0:" in msg and "1:" in msg


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
    from decklight.demos.names import CardNames
    from decklight.engine.duel import Engine

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
    from decklight.engine.actions import Kind

    acts = [[int(Kind.ACTIVATE), -1, code, 0, 0, 0, 0x10, 2, 0, 0] for code in live_codes]
    acts.append([int(Kind.PASS), -1, 0, 0, 0, 0, 0x10, 0, -1, 0])
    g = [0] * 24
    g[2], g[21] = turn, 0x10
    return {
        "actions": acts,
        "action_mask": [1] * len(acts),
        "global": g,
        "chain_mask": [1] * chain_len,
        "cards": [],
        "card_mask": [],
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
        "choices": [],
        "labels": [],
        "actors": [],
        "turns": [],
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
    assert "Battle" in out
    assert "Direct-damage total if the opponent's field were empty:" in out


def test_same_zone_duplicates_get_zone_suffix_and_match():
    import numpy as np

    from decklight.engine.actions import Kind

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
        "cards": cards,
        "card_mask": np.array([True, True]),
    }
    names = {code: "Snake-Eye Ash"}
    disp = hand_play._display_names(obs, names)
    assert disp == ["Snake-Eye Ash (Monster Zone 0)", "Snake-Eye Ash (Monster Zone 2)"], disp
    assert hand_play._match(obs, "SEL:Snake-Eye_Ash_(Monster_Zone_2)", names) == 1


def test_number_options_show_values_and_match():
    from decklight.engine.actions import Kind

    obs = {"actions": [[int(Kind.NUMBER), -1, 0, 0, 0, 5, 143, 0, -1, 0]], "action_mask": [1]}
    assert hand_play._display_names(obs, {}) == ["5"]
    assert hand_play._match(obs, "NUM:5", {}) == 0


@pytest.mark.skipif(
    not any((ROOT / "data" / "demos" / "gold").glob("*.jsonl")),
    reason="gold games absent (data/ is gitignored)",
)
def test_saved_gold_games_replay_unchanged():
    """Every saved gold record replays to the end with all choices legal (hand_play logic unchanged)."""
    import json

    from decklight.demos.dataset import record_config
    from decklight.demos.game import _opts
    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import Engine

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


def _main_phase_battle_obs():
    """Fake Main Phase window: no ATTACK options (they exist only in the Battle Phase), the turn
    player controls two face-up Attack Position monsters and the foe one, plus an Extra Deck
    SPSUMMON on offer (for the summon-then-attack scan)."""
    import numpy as np

    from decklight.engine.actions import Kind

    acts = [
        [int(Kind.SPSUMMON), -1, 89631139, 0, 0, 0, 0xB, 7, 0, 0],  # Blue-Eyes, Extra Deck
        [int(Kind.TO_END), -1, 0, 0, 0, 0, 0xB, 0, -1, 0],
    ]
    cards = np.zeros((3, 21), dtype=np.int32)
    cards[0] = [12345, 3, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2500, 2000, 0, 0, 0, 0, 0]
    cards[1] = [9024367, 3, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2000, 1200, 0, 0, 0, 0, 0]
    cards[2] = [54321, 3, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2000, 1500, 0, 0, 0, 0, 0]
    g = [8000, 3300, 3, 4, 1, 1] + [0] * 18
    g[21] = 0xB  # MSG_SELECT_IDLECMD
    return {
        "actions": acts,
        "action_mask": [1, 1],
        "cards": cards,
        "card_mask": np.array([True, True, True]),
        "global": g,
    }


_BATTLE_NAMES = {
    12345: "Test Attacker",
    9024367: "Galaxy Dragon",
    54321: "Test Foe",
    89631139: "Blue-Eyes White Dragon",
}


def test_battle_helper_lists_main_phase_attackers():
    """The batch-3 defect: `check` in the Main Phase always printed "attackers: none". Attackers now
    come from the seat's face-up Attack Position monsters (estimate, ATTACK options are Battle only)."""
    lines = hand_play._battle_lines(_main_phase_battle_obs(), _BATTLE_NAMES)
    text = "\n".join(lines)
    assert "Your attackers: none" not in text
    assert "Test Attacker [ATK 2500]" in text
    assert "estimate" in lines[0]
    assert "Direct-damage total if the opponent's field were empty: 4500" in text
    assert "Test Attacker -> Test Foe (deals 500)" in text


def test_battle_helper_flags_no_direct_attack_text():
    """Galaxy Dragon's text bars direct attacks: flagged, and the greedy plan holds it."""
    lines = hand_play._battle_lines(_main_phase_battle_obs(), _BATTLE_NAMES)
    text = "\n".join(lines)
    assert "Galaxy Dragon [ATK 2000] (text: cannot attack directly)" in text
    assert "Galaxy Dragon holds (beats nothing face-up)" in text


def test_battle_helper_summon_then_attack():
    """Each SPSUMMON/SUMMON monster on offer lists its ATK plus remaining attackers vs LP."""
    lines = hand_play._battle_lines(_main_phase_battle_obs(), _BATTLE_NAMES)
    text = "\n".join(lines)
    assert "Summon-then-attack (estimate" in text
    assert "Blue-Eyes White Dragon (Extra Deck) [ATK 3000] + remaining attackers 4500" in text
    assert "7500 vs LP 3300 (lethal)" in text


def test_p_never_answers_yes_no_prompts():
    from decklight.engine.actions import Kind

    obs = _fake_obs([(Kind.YES, 0), (Kind.NO, 0)])
    assert hand_play._match(obs, "P", {}) is None
    obs2 = _fake_obs([(Kind.ACTIVATE, 111), (Kind.PASS, 0)])
    assert hand_play._match(obs2, "P", {111: "X"}) == 1


def test_no_match_msg_for_p_on_yes_no_and_kinds():
    from decklight.engine.actions import Kind

    g = [0] * 24
    g[21] = 0xD  # MSG_SELECT_YESNO
    obs = dict(_fake_obs([(Kind.YES, 0), (Kind.NO, 0)]))
    obs["global"] = g
    assert hand_play._no_match_msg(obs, "P") == "!! P does not answer yes/no prompts: use YES or NO"
    g2 = [0] * 24
    g2[21] = 0xB
    obs2 = dict(_fake_obs_loc([(Kind.SUMMON, 111, 2, 0), (Kind.ACTIVATE, 222, 2, 0)]))
    obs2["global"] = g2
    msg = hand_play._no_match_msg(obs2, "BP")
    assert msg == (
        "!! BP: no Battle Phase option at this prompt (MSG_SELECT_IDLECMD); "
        "answer it first (available: SUM, ACT)"
    ), msg


def test_match_ignores_punctuation_and_case():
    from decklight.engine.actions import Kind

    obs = _fake_obs([(Kind.SELECT, 111), (Kind.SELECT, 222), (Kind.SELECT, 333)])
    names = {
        111: "Fiendsmith's Tract",  # apostrophe
        222: "Tenyi Spirit - Adhara",  # hyphen
        333: "D/D Savant Kepler",  # slash
    }
    assert hand_play._match(obs, "SEL:Fiendsmiths_Tract", names) == 0
    assert hand_play._match(obs, "SEL:Tenyi_Spirit_Adhara", names) == 1
    assert hand_play._match(obs, "SEL:D_D_Savant_Kepler", names) == 2
    assert hand_play._match(obs, "SEL:fiendsmiths_tract", names) == 0


def test_pass_star_crosses_seats_with_only_declined_live(tmp_path, monkeypatch):
    """PASS* continues across a seat change while the next seat's window offers no live option
    besides those it already declined this turn (121/206 batch-3 uses stopped at the seat change)."""
    KA, KB = 2526224, 14558127
    hand_play._DECLINED.clear()

    def obs_for(codes):
        return _chain_obs(list(codes))

    live_a = hand_play._live(obs_for([KA]))
    live_b = hand_play._live(obs_for([KB]))
    hand_play._DECLINED[0] = live_a | {("turn", 3)}
    hand_play._DECLINED[1] = live_b | {("turn", 3)}

    end = _chain_obs([KA])
    end["global"][21] = 0xB  # MSG_SELECT_IDLECMD: PASS* stops here

    class _TwoSeatEnv:
        def __init__(self, states, players):
            self.states, self.players = list(states), list(players)
            self.player = self.players.pop(0)

        def step(self, i):
            self.player = self.players.pop(0)
            return self.states.pop(0)

    env = _TwoSeatEnv([obs_for([KB]), obs_for([KA]), end], [0, 1, 0, 0])
    st = {"choices": [], "labels": [], "actors": [], "turns": []}
    _, reason = hand_play._pass_star(env, obs_for([KA]), st, _hold_names([]))
    assert st["labels"] == ["pass", "pass", "pass"]
    assert st["actors"] == [0, 1, 0]  # the seat-1 window was answered, not stopped at
    assert reason.startswith("P0 ") and "not a response window" in reason, reason


def test_log_prints_turn_headers_and_decisions(tmp_path, monkeypatch, capsys):
    """`hand_play.py log <game>`: one turn header with LP plus one line per gold decision."""
    from decklight.demos.names import CardNames
    from decklight.engine.duel import Engine

    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    capsys.readouterr()
    st = hand_play._load("g")
    assert st["choices"] == []
    hand_play._log(st, CardNames(), Engine(), None)  # fresh duel: header-less, no decisions yet
    out = capsys.readouterr().out
    assert "(no decisions yet)" in out
    # one real decision, then the log shows a header plus the T1 line (read-only: no state change)
    engine = Engine()
    env, obs = hand_play._replay(engine, st)
    st["choices"].append(0)
    st["labels"].append("gold")
    st["actors"].append(env.player)
    st["turns"].append(int(obs["global"][2]))
    hand_play._log(st, CardNames(), engine, None)
    out = capsys.readouterr().out
    assert "== Turn 1 — LP P0 8000, P1 8000" in out
    assert "T1 P0 " in out
    assert hand_play._load("g")["choices"] == []  # log changed no state


def test_deck_sheet_brief_truncates_text():
    import importlib.util

    spec = importlib.util.spec_from_file_location("deck_sheet", ROOT / "scripts" / "deck_sheet.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["deck_sheet"] = mod
    spec.loader.exec_module(mod)
    full = mod.sheet("snake-eye")
    brief = mod.sheet("snake-eye", brief=True)
    assert len(brief) < len(full)
    for line in brief.splitlines():
        if line.startswith("== "):
            continue
        text = line.split(" — ", 1)[1]
        assert len(text) <= 300, line[:80]
    assert brief.splitlines()[0].startswith("== Main")


def _idle_obs_with(items, msg=0xB, turn=3):
    """Fake window with full 10-col option rows: items = (kind, code, board_row, arg)."""
    from decklight.engine.actions import Kind  # noqa: F401

    actions = [[int(k), row, code, 0, 0, arg, 0, 0, -1, 0] for k, code, row, arg in items]
    g = [0] * 24
    g[2], g[21] = turn, msg
    return {"actions": actions, "action_mask": [1] * len(actions), "global": g}


def test_battle_lines_use_probed_attackers_not_text():
    """`check` battle section comes from the probed ATTACK options: a monster whose own text is
    clean but whose engine direct flag is 0 is flagged engine-true (u33 Cooky class)."""
    import numpy as np

    from decklight.engine.actions import Kind

    code = 12345
    cards = np.zeros((1, 21), dtype=np.int32)
    cards[0, 0], cards[0, 1], cards[0, 2] = code, 3, 1
    cards[0, 3], cards[0, 5], cards[0, 14] = 0, 1, 1000
    obs = _idle_obs_with([(Kind.SPSUMMON, 999, -1, 0)])
    obs["cards"] = cards
    obs["card_mask"] = np.array([True])
    lines = hand_play._battle_lines(obs, {code: "Cooky", 999: "Other"}, [(code, 1000, False)])
    text = "\n".join(lines)
    assert text.startswith("Battle (engine:")
    assert "Cooky [ATK 1000] (engine: cannot attack directly)" in text


def test_probe_returns_none_with_no_battle_option(tmp_path, monkeypatch):
    """First-turn graceful path: a fresh duel (turn 1, no Battle Phase) probes to None."""
    from decklight.engine.duel import Engine

    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    st = hand_play._load("g")
    assert hand_play._probe_battle_attackers(Engine(), st) is None


def test_check_uses_engine_attacker_list(tmp_path, monkeypatch, capsys):
    """`check` prints the probed attacker list (monkeypatched probe)."""
    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    capsys.readouterr()
    monkeypatch.setattr(hand_play, "_probe_for_check", lambda engine, obs, st: [(424, 1000, False)])
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "check", "g"])
    hand_play.main()
    out = capsys.readouterr().out
    assert "Battle (engine:" in out
    assert "(engine: cannot attack directly)" in out


def _facedown_obs():
    """Two SELECT options: a set S/T (face-down) and a face-up monster."""
    import numpy as np

    from decklight.engine.actions import Kind

    cards = np.zeros((2, 21), dtype=np.int32)
    cards[0, 0], cards[0, 1], cards[0, 2] = 1111, 4, 1
    cards[0, 3], cards[0, 5] = 0, 6  # own set spell/trap
    cards[1, 0], cards[1, 1], cards[1, 2] = 2222, 3, 1
    cards[1, 3], cards[1, 5], cards[1, 14] = 0, 1, 2500
    g = [0] * 24
    g[21] = 0xF  # MSG_SELECT_CARD
    return {
        "actions": [
            [int(Kind.SELECT), 0, 1111, 0, 0, 0, 0, 4, 0, 0],
            [int(Kind.SELECT), 1, 2222, 0, 0, 0, 0, 3, 0, 0],
        ],
        "action_mask": [1, 1],
        "cards": cards,
        "card_mask": np.array([True, True]),
        "global": g,
    }


def test_sel_face_down_picks_unique():
    names = {1111: "Infinite Impermanence", 2222: "Blue-Eyes White Dragon"}
    for tok in ("SEL:face-down", "SEL:opponents_face-down", "SEL:set", "SEL:facedown"):
        assert hand_play._match(_facedown_obs(), tok, names) == 0, tok


def test_sel_face_down_lists_several(capsys):
    obs = _facedown_obs()
    cards = obs["cards"].copy()
    cards[1, 1], cards[1, 5] = 4, 6  # both face-down now
    obs["cards"] = cards
    names = {1111: "Infinite Impermanence", 2222: "Infinite Impermanence"}
    assert hand_play._match(obs, "SEL:face-down", names) is None
    msg = hand_play._no_match_msg(obs, "SEL:face-down", names)
    assert msg.startswith("!! token 'SEL:face-down' matches several face-down cards"), msg
    assert "0:" in msg and "1:" in msg


def test_cancel_words_answer_passive():
    from decklight.engine.actions import Kind

    g = [0] * 24
    g[21] = 0xF  # MSG_SELECT_CARD, not a yes/no prompt
    obs = dict(_fake_obs([(Kind.SELECT, 111), (Kind.CANCEL, 0)]))
    obs["global"] = g
    assert hand_play._match(obs, "CAN", {}) == 1
    assert hand_play._match(obs, "CANCEL", {}) == 1
    assert hand_play._match(obs, "NO", {}) == 1
    assert hand_play._match(obs, "CAN:cancel", {}) == 1
    # at a YES/NO prompt NO is still a real decision and P still refuses
    g2 = [0] * 24
    g2[21] = 0xD
    obs2 = dict(_fake_obs([(Kind.YES, 0), (Kind.NO, 0)]))
    obs2["global"] = g2
    assert hand_play._match(obs2, "NO", {}) == 1
    assert hand_play._match(obs2, "P", {}) is None


def test_commands_are_case_insensitive(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "CHECK", "g"])
    hand_play.main()
    assert "Unused options:" in capsys.readouterr().out


def test_bp_no_match_names_the_prompt():
    from decklight.engine.actions import Kind

    g = [0] * 24
    g[21] = 0x10  # MSG_SELECT_CHAIN
    obs = dict(_fake_obs([(Kind.ACTIVATE, 111), (Kind.PASS, 0)]))
    obs["global"] = g
    msg = hand_play._no_match_msg(obs, "BP")
    assert msg.startswith("!! BP: no Battle Phase option at this prompt (MSG_SELECT_CHAIN)"), msg


def _battle_obs(actions, card_rows, msg=0xA, turn=3, phase=0x40, my_turn=1):
    """Fake battle-phase window: actions = (kind, code, board_row, arg); card_rows = 21-col rows."""
    import numpy as np

    cards = np.zeros((len(card_rows), 21), dtype=np.int32)
    for r, row in enumerate(card_rows):
        cards[r] = row
    acts = [[int(k), row, code, 0, 0, arg, 0, 3, 0, 0] for k, code, row, arg in actions]
    g = [8000, 4000, turn, phase, 0, my_turn] + [0] * 18
    g[21] = msg
    return {
        "actions": acts,
        "action_mask": [1] * len(acts),
        "cards": cards,
        "card_mask": np.array([True] * len(card_rows)),
        "global": g,
        "chain_mask": [],
        "chain": [],
    }


def _mon_row(code, seq, atk):
    import numpy as np

    row = np.zeros(21, dtype=np.int32)
    row[0], row[1], row[2] = code, 3, seq
    row[3], row[5], row[14], row[15] = 0, 1, atk, 1000
    return row


def test_multi_attack_tokens_bridge_chain_windows(tmp_path, monkeypatch, capsys):
    """Several ATK: tokens in one `do`: after each attack the empty chain windows on the way back
    to MSG_SELECT_BATTLECMD are passed (like --auto-pass); both attacks land in one command."""
    from decklight.demos.names import CardNames
    from decklight.engine.actions import Kind

    hand_play._DECLINED.clear()
    codes = CardNames()
    cA = next(c for c, n in codes.names.items() if n == "Blue-Eyes White Dragon")
    cB = next(c for c, n in codes.names.items() if n == "Dark Magician")
    nA = codes.names[cA].replace(" ", "_")
    nB = codes.names[cB].replace(" ", "_")
    obs0 = _battle_obs(
        [(Kind.ATTACK, cA, 0, 1), (Kind.TO_END, 0, -1, 0)],
        [_mon_row(cA, 1, 3000), _mon_row(cB, 2, 2500)],
    )
    chain_obs = _battle_obs([(Kind.PASS, 0, -1, 0)], [], msg=0x10, my_turn=0)
    obs1 = _battle_obs(
        [(Kind.ATTACK, cB, 1, 1), (Kind.TO_END, 0, -1, 0)],
        [_mon_row(cA, 1, 3000), _mon_row(cB, 2, 2500)],
    )
    idle_obs = _battle_obs([(Kind.TO_END, 0, -1, 0)], [_mon_row(cB, 2, 2500)], msg=0xB)

    class _SeqEnv:
        select_hint = 0
        winner = None

        def __init__(self):
            self.player = 0
            self.script = [(1, chain_obs), (0, obs1), (0, idle_obs)]

        def step(self, i):
            self.player, obs = self.script.pop(0)
            return obs

    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    capsys.readouterr()
    monkeypatch.setattr(hand_play, "_replay", lambda engine, st: (_SeqEnv(), obs0))
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "do", "g", f"ATK:{nA} ATK:{nB}"])
    hand_play.main()
    out = capsys.readouterr().out
    assert "attack bridge passed 1 windows" in out
    st = hand_play._load("g")
    assert st["labels"] == ["gold", "pass", "gold"], st["labels"]
    assert st["actors"] == [0, 1, 0], st["actors"]


def test_material_lines_show_zone_and_link_rating(tmp_path, monkeypatch, capsys):
    """Link-material SELECT/UNSELECT lines name the card with its zone plus Link rating."""
    import numpy as np

    from decklight.demos.names import CardNames
    from decklight.engine.actions import Kind

    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    st = hand_play._load("g")
    capsys.readouterr()
    codes = CardNames()
    sp = next(c for c, n in codes.names.items() if n == "S:P Little Knight")
    baron = next(c for c, n in codes.names.items() if "Barong" in n)
    cards = np.zeros((2, 21), dtype=np.int32)
    cards[0, 0], cards[0, 1], cards[0, 2] = baron, 3, 1
    cards[0, 3], cards[0, 5], cards[0, 14] = 0, 1, 1800
    cards[1, 0], cards[1, 1], cards[1, 2] = sp, 3, 6
    cards[1, 3], cards[1, 5], cards[1, 14] = 0, 1, 1600
    g = [8000, 8000, 2, 0x100, 1, 1] + [0] * 18
    g[21] = 0x1A  # MSG_SELECT_UNSELECT_CARD
    obs = {
        "actions": [
            [int(Kind.SELECT), 0, baron, 0, 0, 0, 0x1A, 3, 0, 0],
            [int(Kind.SELECT), 1, sp, 0, 0, 0, 0x1A, 3, 0, 0],
        ],
        "action_mask": [1, 1],
        "cards": cards,
        "card_mask": np.array([True, True]),
        "global": g,
        "chain_mask": [],
        "chain": [],
    }

    class _Env:
        player = 0
        select_hint = 0

    hand_play._show(_Env(), obs, st, codes, False)
    out = capsys.readouterr().out
    assert "Monster Zone" in out
    assert "[Link-2]" in out
    assert hand_play._link_rating(sp) == 2
    assert hand_play._link_rating(baron) is None


def _zone_obs():
    """SELECT prompt over four board cards: own set S/T (S0), opp set S/T (S1),
    own face-up monster (M0), opp face-down monster (M2)."""
    import numpy as np

    from decklight.engine.actions import Kind

    cards = np.zeros((4, 21), dtype=np.int32)
    cards[0, 0], cards[0, 1], cards[0, 2] = 1111, 4, 1
    cards[0, 3], cards[0, 5] = 0, 6  # own set spell/trap
    cards[1, 0], cards[1, 1], cards[1, 2] = 0, 4, 2
    cards[1, 3], cards[1, 5] = 1, 6  # opponent's set spell/trap
    cards[2, 0], cards[2, 1], cards[2, 2] = 2222, 3, 1
    cards[2, 3], cards[2, 5], cards[2, 14] = 0, 1, 2500  # own face-up monster
    cards[3, 0], cards[3, 1], cards[3, 2] = 0, 3, 3
    cards[3, 3], cards[3, 5] = 1, 4  # opponent's face-down monster
    g = [0] * 24
    g[21] = 0xF  # MSG_SELECT_CARD
    return {
        "actions": [
            [int(Kind.SELECT), 0, 1111, 0, 0, 0, 0, 4, 0, 0],
            [int(Kind.SELECT), 1, 0, 0, 0, 0, 0, 4, 1, 0],
            [int(Kind.SELECT), 2, 2222, 0, 0, 0, 0, 3, 0, 0],
            [int(Kind.SELECT), 3, 0, 0, 0, 0, 0, 3, 1, 0],
        ],
        "action_mask": [1, 1, 1, 1],
        "cards": cards,
        "card_mask": np.array([True] * 4),
        "global": g,
    }


_ZONE_NAMES = {1111: "Infinite Impermanence", 2222: "Blue-Eyes White Dragon"}


def test_sel_controller_qualifier_picks_face_down():
    obs = _zone_obs()
    assert hand_play._match(obs, "SEL:own:face-down", _ZONE_NAMES) == 0
    assert hand_play._match(obs, "SEL:opp:M2", _ZONE_NAMES) == 3
    assert hand_play._match(obs, "SEL:opp:S1", _ZONE_NAMES) == 1
    assert hand_play._match(obs, "SEL:own:S0", _ZONE_NAMES) == 0
    assert hand_play._match(obs, "SEL:S1", _ZONE_NAMES) == 1  # bare zone token
    assert hand_play._match(obs, "SEL:M0", _ZONE_NAMES) == 2
    assert hand_play._match(obs, "SEL:opp:face-down", _ZONE_NAMES) is None  # two opp face-downs
    assert hand_play._match(obs, "SEL:own:M2", _ZONE_NAMES) is None  # nothing there


def test_sel_ambiguous_lists_controller_and_zone():
    obs = _zone_obs()
    assert hand_play._match(obs, "SEL:face-down", _ZONE_NAMES) is None
    msg = hand_play._no_match_msg(obs, "SEL:face-down", _ZONE_NAMES)
    assert msg.startswith("!! token 'SEL:face-down' matches several face-down cards"), msg
    for i in ("0:", "1:", "3:"):
        assert i in msg, msg
    assert "opponent's" in msg and "your" in msg  # controllers
    assert "Spell/Trap Zone" in msg and "Monster Zone" in msg  # zones
    assert "face-down" in msg


def test_sel_ambiguous_names_stop_but_identical_copies_pick_first():
    from decklight.engine.actions import Kind

    obs = _fake_obs([(Kind.SELECT, 111), (Kind.SELECT, 222)])
    names = {111: "Mirror Force", 222: "Mirror Gate"}
    assert hand_play._match(obs, "SEL:Mirror", names) is None  # same-rank tie, different cards
    msg = hand_play._no_match_msg(obs, "SEL:Mirror", names)
    assert "matches several options" in msg and "0:" in msg and "1:" in msg, msg
    # identical copies (same code, controller, location kind) stay interchangeable
    obs2 = _fake_obs_loc([(Kind.SELECT, 111, 2, 0), (Kind.SELECT, 111, 2, 0)])
    assert hand_play._match(obs2, "SEL:Mirror_Force", {111: "Mirror Force"}) == 0


def _sel_card_obs():
    """MSG_SELECT_CARD with two SELECT options and no ATTACK option."""
    import numpy as np

    from decklight.engine.actions import Kind

    cards = np.zeros((2, 21), dtype=np.int32)
    cards[0, 0], cards[0, 1], cards[0, 2] = 2222, 3, 1
    cards[0, 3], cards[0, 5], cards[0, 14] = 0, 1, 2500
    cards[1, 0], cards[1, 1], cards[1, 2] = 3333, 3, 2
    cards[1, 3], cards[1, 5], cards[1, 14] = 0, 1, 2000
    g = [0] * 24
    g[21] = 0xF  # MSG_SELECT_CARD
    obs = {
        "actions": [
            [int(Kind.SELECT), 0, 2222, 0, 0, 0, 0xF, 3, 0, 0],
            [int(Kind.SELECT), 1, 3333, 0, 0, 0, 0xF, 3, 0, 0],
        ],
        "action_mask": [1, 1],
        "cards": cards,
        "card_mask": np.array([True, True]),
        "global": g,
    }
    return obs, {2222: "Blue-Eyes White Dragon", 3333: "Dark Magician"}


def test_atk_at_selection_applies_unique():
    obs, names = _sel_card_obs()
    idx, note = hand_play._atk_fallback(obs, "ATK:Blue-Eyes_White_Dragon", names)
    assert idx == 0
    assert note == "(ATK at a selection prompt: applied as SEL:Blue-Eyes_White_Dragon)"


def test_atk_at_selection_stops_when_ambiguous():
    obs, _ = _sel_card_obs()
    names = {2222: "Test Dragon A", 3333: "Test Dragon B"}
    idx, msg = hand_play._atk_fallback(obs, "ATK:Dragon", names)
    assert idx is None
    assert "card selection" in msg and "0:" in msg and "1:" in msg, msg
    idx, msg = hand_play._atk_fallback(obs, "ATK:Blue-Eyes", names)
    assert idx is None and "card selection" in msg  # matches nothing


def test_atk_fallback_not_applicable_elsewhere():
    from decklight.engine.actions import Kind

    obs, names = _sel_card_obs()
    obs["global"][21] = 0xB  # MSG_SELECT_IDLECMD: not a selection prompt
    assert hand_play._atk_fallback(obs, "ATK:Blue-Eyes_White_Dragon", names) is None
    obs["global"][21] = 0xF
    obs["actions"].append([int(Kind.ATTACK), 0, 2222, 0, 0, 1, 0xA, 3, 0, 0])
    obs["action_mask"] = [1, 1, 1]
    assert hand_play._atk_fallback(obs, "ATK:Blue-Eyes_White_Dragon", names) is None
    assert hand_play._atk_fallback(obs, "SEL:Blue-Eyes_White_Dragon", names) is None  # not an ATK token


def test_do_applies_atk_as_sel_with_note(tmp_path, monkeypatch, capsys):
    """End-to-end: ATK:<card> at a selection prompt applies the SEL and prints the note."""
    obs, names = _sel_card_obs()

    class _Env:
        player = 0
        select_hint = 0
        winner = None

        def step(self, i):
            assert i == 0
            # duel over after the pick: fall off the end (returns None)

    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    capsys.readouterr()
    monkeypatch.setattr(hand_play, "_replay", lambda engine, st: (_Env(), obs))
    monkeypatch.setattr(hand_play, "CardNames", lambda: type("N", (), {"names": names})())
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "do", "g", "ATK:Blue-Eyes_White_Dragon"])
    hand_play.main()
    out = capsys.readouterr().out
    assert "(ATK at a selection prompt: applied as SEL:Blue-Eyes_White_Dragon)" in out, out
    st = hand_play._load("g")
    assert st["choices"] == [0] and st["labels"] == ["gold"], st
