"""Games v2: named sessions, plan-item serde + dispatch, Interrupter. No network, LLM, engine search or vast."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from decklight.decks import Deck
from decklight.demos.schema import ComboLine, Step
from decklight.demos.verify import Verdict
from decklight.engine import constants as C
from decklight.engine.actions import Kind
from decklight.engine.duel import DuelConfig
from decklight.infra import remote, vast

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- named sessions ---


def test_session_file_naming(tmp_path, monkeypatch):
    monkeypatch.setattr(vast, "ROOT", tmp_path)
    monkeypatch.setattr(vast, "SESSION_FILE", tmp_path / "data" / "vast_session.json")
    assert vast.session_file("lines") == tmp_path / "data" / "vast_session.json"
    assert vast.session_file("games") == tmp_path / "data" / "vast_session-games.json"


def test_destroy_strays_spares_both_sessions(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(vast, "ROOT", tmp_path)
    monkeypatch.setattr(vast, "SESSION_FILE", tmp_path / "data" / "vast_session.json")
    monkeypatch.setattr(vast.time, "sleep", lambda s: None)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "vast_session.json").write_text(json.dumps({"instance_id": 101}))
    (tmp_path / "data" / "vast_session-games.json").write_text(json.dumps({"instance_id": 102}))
    live = {101: "ygo-zero-verify-0000", 102: "ygo-zero-games-0000", 103: "ygo-zero-stray"}
    destroyed = []

    def fake_cli(*args, raw=True, timeout=120):
        if args[:2] == ("show", "instances"):
            return [{"id": i, "label": lab} for i, lab in live.items()]
        if args[:2] == ("destroy", "instance"):
            destroyed.append(int(args[2]))
            live.pop(int(args[2]), None)
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(vast, "cli", fake_cli)
    assert vast.destroy_strays() == [103]
    assert destroyed == [103]


# --- plan items ---

_DECK = Deck((89631139,), (), (), "d")


def _plan_parts():
    cfg = DuelConfig(Deck((89631139,), (), (), "d0"), Deck((8240199,), (), (), "d1"), seed=3)
    line = ComboLine(
        id="t1",
        deck="d",
        hand=["Blue-Eyes White Dragon"],
        hand_codes=[89631139],
        steps=[Step(i=0, action="normal_summon", card="Blue-Eyes White Dragon", card_code=89631139)],
        going="first",
    )
    return cfg, [0, 2], line, _DECK, 500


def test_plan_item_serde_roundtrip(tmp_path):
    cfg, prefix, line, deck, budget = _plan_parts()
    item = remote.plan_item_to_json(cfg, prefix, line, deck, budget)
    assert item["kind"] == "plan"
    p = tmp_path / "items.jsonl"
    remote.write_items(p, [item])
    (back,) = remote.read_items(p)
    assert back == item


def test_plan_item_mixed_with_legacy_tuples(tmp_path):
    cfg, prefix, line, deck, budget = _plan_parts()
    plan = remote.plan_item_to_json(cfg, prefix, line, deck, budget)
    legacy = (line.to_json(), (89631139,), (), (), "d", 0, 100)
    p = tmp_path / "items.jsonl"
    remote.write_items(p, [legacy, plan])
    back = remote.read_items(p)
    assert back[0] == legacy
    assert back[1] == plan


def test_verify_items_dispatch_on_kind(monkeypatch):
    vi = _load("verify_items")
    cfg, prefix, line, deck, budget = _plan_parts()
    item = remote.plan_item_to_json(cfg, prefix, line, deck, budget)
    seen = {}

    def fake_plan(engine, config, pfx, ln, dk, budget=4000):
        seen["args"] = (config, pfx, ln, dk, budget)
        return Verdict("t1", True, "ok", [], [], 3, 1, [], "ok", [])

    monkeypatch.setattr(vi, "plan_from_state", fake_plan)
    v = vi._dispatch(item)
    assert v.ok and v.line_id == "t1"
    config, pfx, ln, dk, b = seen["args"]
    assert (pfx, b) == (prefix, budget)
    assert config.seed == 3 and dk == deck and ln.id == "t1"


# --- Interrupter ---


def _fake_obs(my_turn=0, msg=C.MSG_SELECT_CHAIN, kinds=((Kind.ACTIVATE, 89631139), (Kind.PASS, 0))):
    g = np.zeros(24, dtype=np.int32)
    g[2], g[3], g[5], g[21] = 3, 0x04, my_turn, msg
    acts = np.zeros((64, 10), dtype=np.int32)
    amask = np.zeros(64, bool)
    for i, (k, code) in enumerate(kinds):
        acts[i, 0], acts[i, 2] = int(k), code
        amask[i] = True
    return {
        "global": g,
        "cards": np.zeros((160, 21), dtype=np.int32),
        "card_mask": np.zeros(160, bool),
        "chain": np.zeros((16, 5), dtype=np.int32),
        "chain_mask": np.zeros(16, bool),
        "actions": acts,
        "action_mask": amask,
    }


def _interrupter(answer, **kw):
    from decklight.demos.game import Interrupter

    calls = []
    it = Interrupter(lambda p: calls.append(p) or answer, {89631139: "Blue-Eyes White Dragon"}, **kw)
    return it, calls


def test_interrupter_valid_answer():
    it, _ = _interrupter('{"choice": 0, "why": "stops the search"}')
    assert it.wants(_fake_obs())
    assert it(_fake_obs(), "sky-striker", 3, 1, ["Ash Blossom"], None) == (0, "stops the search")


def test_interrupter_invalid_answer_passes():
    it, _ = _interrupter("not json at all")
    assert it(_fake_obs(), "sky-striker", 3, 1, [], None)[0] == 1  # the PASS index
    it2, _ = _interrupter('{"choice": 99, "why": "bad"}')
    assert it2(_fake_obs(), "sky-striker", 3, 1, [], None)[0] == 1
    it3, _ = _interrupter('```json\n{"choice": -1, "why": "wait"}\n```')
    assert it3(_fake_obs(), "sky-striker", 3, 1, [], None) == (1, "wait")


def test_interrupter_cap_per_turn():
    it, calls = _interrupter('{"choice": 0, "why": "go"}', max_per_turn=1)
    assert it(_fake_obs(), "d", 3, 1, [], None) == (0, "go")
    assert it(_fake_obs(), "d", 3, 1, [], None) == (1, "no-ask")
    assert len(calls) == 1
    assert it(_fake_obs(), "d", 4, 1, [], None) == (0, "go")  # a new turn gets fresh asks


def test_interrupter_only_on_opponent_turn():
    from decklight.engine.actions import Kind as K

    it, calls = _interrupter('{"choice": 0, "why": "go"}')
    own = _fake_obs(my_turn=1)  # acting player IS the turn player
    assert not it.wants(own)
    assert it(own, "d", 3, 0, [], None) == (1, "no-ask")  # passive (PASS) index
    no_act = _fake_obs(kinds=((K.PASS, 0),))
    assert not it.wants(no_act)
    assert calls == []


def test_legal_actions_lists_engine_options_and_flags_live_plays():
    from decklight.demos.game import legal_actions, turn_prompt

    names = {1: "Pre-Preparation of Rites", 2: "Ragged", 3: "Trap A"}
    obs = _fake_obs(1, C.MSG_SELECT_IDLECMD, ((Kind.ACTIVATE, 1), (Kind.ACTIVATE, 1), (Kind.SUMMON, 2),
                                              (Kind.SSET, 3), (Kind.TO_BATTLE, 0), (Kind.TO_END, 0)))
    text, live = legal_actions(obs, names)
    assert text == "Activate: Pre-Preparation of Rites; Normal Summon: Ragged; Set spell/trap: Trap A"
    assert live
    text, live = legal_actions(_fake_obs(1, C.MSG_SELECT_IDLECMD, ((Kind.SSET, 3), (Kind.TO_END, 0))), names)
    assert text == "Set spell/trap: Trap A" and not live
    assert "LEGAL RIGHT NOW" in turn_prompt("x", "", "Your hand: none", True, [], text)
    assert "LEGAL RIGHT NOW" not in turn_prompt("x", "", "Your hand: none", True, [])


def test_step_player_asks_only_real_choices_and_logs_the_turn():
    from decklight.demos.game import StepPlayer

    asked = []

    def llm(prompt):
        asked.append(prompt)
        return '{"choice": 0, "why": "search first"}'

    sp = StepPlayer(llm, {1: "Pre-Preparation of Rites"}, max_per_turn=2)
    idle = _fake_obs(1, C.MSG_SELECT_IDLECMD, ((Kind.ACTIVATE, 1), (Kind.TO_END, 0)))
    assert sp(idle, "deck", 3) == (0, "search first")
    assert "THIS TURN SO FAR: nothing yet" in asked[0]
    sp(idle, "deck", 3)
    assert "ACTIVATE Pre-Preparation of Rites" in asked[1]
    assert sp(idle, "deck", 3) is None  # asks exhausted: the rule decides
    one_target = _fake_obs(1, C.MSG_SELECT_CARD, ((Kind.SELECT, 1), (Kind.CANCEL, 0)))
    assert not sp.wants(one_target)
    place = _fake_obs(1, C.MSG_SELECT_PLACE, ((Kind.PLACE, 0), (Kind.PLACE, 0)))
    assert not sp.wants(place)
