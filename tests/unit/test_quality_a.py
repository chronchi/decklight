"""Task A: strategy notes in prompts, best-of-N planning, --session plumbing.

No network/LLM/vast: the LLM and verifier are faked, remote_verify is
monkeypatched.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from decklight.decks import Deck
from decklight.demos import strategy
from decklight.demos.names import CardNames
from decklight.demos.plan import (
    improve_prompt,
    plan_hands,
    plan_prompt,
    select_best,
)
from decklight.demos.schema import ComboLine
from decklight.demos.verify import Verdict

ROOT = Path(__file__).resolve().parents[2]

NAMES = CardNames()
SAGE, PWWB, BLUE_EYES = 8240199, 63356631, 89631139
SAGE_N, PWWB_N, BEWD_N = "Sage with Eyes of Blue", "Phoenix Wing Wind Blast", "Blue-Eyes White Dragon"
FAKE_ENGINE = SimpleNamespace(cards={})


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _deck() -> Deck:
    main = (SAGE, PWWB, BLUE_EYES, SAGE, PWWB, BLUE_EYES, 48800175, 97268402, 38517737, 89631139)
    return Deck(main, (), (), "test-deck")


def _cand(title: str, end: list[str]) -> dict:
    steps = [
        {"i": 0, "action": "normal_summon", "card": SAGE_N, "text": "x"},
        {"i": 1, "action": "set_spelltrap", "card": PWWB_N, "text": "y"},
    ]
    return {"title": title, "hand": ["PLACEHOLDER"], "steps": steps, "end_board": end, "going": "first"}


def _resp(*cands: dict) -> str:
    return json.dumps({"lines": list(cands)})


def _verdict(ok: bool, line_id: str, executed: int = 2, reason: str = "ok") -> Verdict:
    v = Verdict(line_id, ok, reason, [0], ["stated", "auto"], 0, 3, [], f"{reason} detail", [])
    v.executed_steps = executed
    return v


# --- notes injection ---


def test_notes_absent_prompts_have_no_strategy_block(monkeypatch, tmp_path):
    monkeypatch.setattr(strategy, "PRINCIPLES_PATH", tmp_path / "none.md")
    assert strategy.notes_block("no-such-deck-xyz") == ""
    p = plan_prompt("no-such-deck-xyz", "SHEET", ["A"], [])
    assert "STRATEGY" not in p
    q = improve_prompt("no-such-deck-xyz", "SHEET", ["A"], "{}", ["X"])
    assert "STRATEGY" not in q


def test_principles_go_into_every_notes_block(monkeypatch, tmp_path):
    f = tmp_path / "PRINCIPLES.md"
    f.write_text("GENERAL PRINCIPLES\n- attack for lethal")
    monkeypatch.setattr(strategy, "PRINCIPLES_PATH", f)
    assert "attack for lethal" in strategy.notes_block("no-such-deck-xyz")
    assert "attack for lethal" in plan_prompt("no-such-deck-xyz", "SHEET", ["A"], [])


def test_notes_present_injected_after_decklist(tmp_path, monkeypatch):
    (tmp_path / "note-deck.md").write_text("Hold the Ash for Maxx C.\n")
    monkeypatch.setattr(strategy, "NOTES_DIR", tmp_path)
    p = plan_prompt("note-deck", "SHEET-ABC", ["A"], [])
    assert "STRATEGY NOTES for note-deck" in p and "Hold the Ash" in p
    assert p.index("SHEET-ABC") < p.index("STRATEGY NOTES") < p.index("OPENING HAND")
    q = improve_prompt("note-deck", "SHEET-ABC", ["A"], "{}", ["X"])
    assert "STRATEGY NOTES for note-deck" in q
    assert q.index("SHEET-ABC") < q.index("STRATEGY NOTES") < q.index("OPENING HAND")


def test_notes_absent_prompts_unchanged_for_cache():
    # Default candidates=1 without notes must match the old explicit call shape.
    a = plan_prompt("no-such-deck-xyz", "SHEET", ["A", "B"], [])
    b = plan_prompt("no-such-deck-xyz", "SHEET", ["A", "B"], [], candidates=1)
    assert a == b
    assert "DIFFERENT" not in a


# --- best-of-N ---


def test_plan_prompt_candidates_asks_for_different_lines():
    p = plan_prompt("d", "sheet", ["A"], [], candidates=3)
    assert "3" in p and "DIFFERENT" in p and '"lines"' in p
    assert "route or end board" in p


def test_best_of_n_selects_biggest_board():
    deck = _deck()
    hands = [[SAGE, PWWB, BLUE_EYES, SAGE, PWWB]]
    resp = _resp(
        _cand("small", [SAGE_N]),
        _cand("big", [SAGE_N, PWWB_N, BEWD_N]),
        _cand("mid", [SAGE_N, PWWB_N]),
    )

    def fake_verify(items):
        return [_verdict(True, ln.id, executed=2) for ln, _, _ in items]

    res = plan_hands(
        FAKE_ENGINE, deck, "test-deck", hands, [], NAMES,
        lambda prompts: [resp], fake_verify, repairs=0, candidates=3,
    )
    r = res[0]
    assert r["ok_line"] is not None and r["ok_line"].title == "big"
    assert r["ok_line"].id.endswith("-c1")
    # Distinct extras: mid and small both differ from big.
    assert {x["line"].title for x in r["extras"]} == {"small", "mid"}
    assert len(r["attempts"]) == 3  # all candidates recorded


def test_best_of_n_tie_goes_to_more_executed_steps():
    deck = _deck()
    hands = [[SAGE, PWWB, BLUE_EYES, SAGE, PWWB]]
    resp = _resp(_cand("a", [SAGE_N, PWWB_N]), _cand("b", [SAGE_N, BEWD_N]))

    def fake_verify(items):
        ids = [ln.id for ln, _, _ in items]
        assert ids[0].endswith("-c0") and ids[1].endswith("-c1")
        return [_verdict(True, ln.id, executed=2) for ln, _, _ in items[:1]] + [
            _verdict(True, items[1][0].id, executed=9)
        ]

    res = plan_hands(
        FAKE_ENGINE, deck, "test-deck", hands, [], NAMES,
        lambda prompts: [resp], fake_verify, repairs=0, candidates=2,
    )
    assert res[0]["ok_line"].title == "b"  # same size, more steps wins


def test_best_of_n_same_board_excluded_from_extras():
    deck = _deck()
    hands = [[SAGE, PWWB, BLUE_EYES, SAGE, PWWB]]
    resp = _resp(
        _cand("best", [SAGE_N, PWWB_N, BEWD_N]),
        _cand("same-board", [SAGE_N, PWWB_N, BEWD_N]),
        _cand("other", [SAGE_N]),
    )

    def fake_verify(items):
        return [_verdict(True, ln.id, executed=2) for ln, _, _ in items]

    res = plan_hands(
        FAKE_ENGINE, deck, "test-deck", hands, [], NAMES,
        lambda prompts: [resp], fake_verify, repairs=0, candidates=3,
    )
    assert res[0]["ok_line"].title == "best"  # tie at 3 goes to the first
    assert [x["line"].title for x in res[0]["extras"]] == ["other"]


def test_select_best_scores_and_distinct_rule():
    from decklight.demos.schema import Step

    def line(end):
        ln = ComboLine(id="x", deck="d", source_file="p", hand=[], end_board=list(end), going="first")
        ln.steps = [Step(i=0, action="normal_summon", card=SAGE_N, text="x")]
        ln.end_board_codes = [1] * len(end)
        return ln

    cands = [(line(["a", "b"]), _verdict(True, "i0", executed=1)), (line(["a"]), _verdict(True, "i1", executed=9))]
    best_i, extras = select_best(cands)
    assert best_i == 0 and extras == [cands[1]]  # bigger wins despite fewer steps
    same = [(line(["a"]), _verdict(True, "i0", executed=1)), (line(["a"]), _verdict(True, "i1", executed=5))]
    best_i, extras = select_best(same)
    assert best_i == 1 and extras == []  # tie -> more steps; same board -> no extras


def test_best_of_n_repairs_first_when_none_verified():
    deck = _deck()
    hands = [[SAGE, PWWB, BLUE_EYES, SAGE, PWWB]]
    seen: list[str] = []
    calls = {"n": 0}

    def fake_complete(prompts):
        calls["n"] += 1
        seen.extend(prompts)
        if calls["n"] == 1:
            return [_resp(_cand("first-cand", [SAGE_N]), _cand("second-cand", [PWWB_N]))]
        return [_resp(_cand("repaired", [SAGE_N, PWWB_N]))]

    def fake_verify(items):
        out = []
        for ln, _, _ in items:
            if ln.id.endswith("-a0-c0") or ln.id.endswith("-a0-c1"):
                out.append(_verdict(False, ln.id, reason="no_anchor@1"))
            else:
                out.append(_verdict(True, ln.id))
        return out

    res = plan_hands(
        FAKE_ENGINE, deck, "test-deck", hands, [], NAMES,
        fake_complete, fake_verify, repairs=1, candidates=2,
    )
    assert calls["n"] == 2  # round 0 + one repair (no repair when something verified)
    assert "engine rejected it" in seen[1]
    assert "first-cand" in seen[1]  # repairs the FIRST candidate
    assert len(res[0]["attempts"]) == 3  # 2 candidates + 1 repair
    assert res[0]["ok_line"] is not None and res[0]["ok_line"].id.endswith("-a1")


def test_candidates_one_keeps_single_line_behaviour():
    deck = _deck()
    hands = [[SAGE, PWWB, BLUE_EYES, SAGE, PWWB]]
    resp = _resp(_cand("only", [SAGE_N]))

    def fake_verify(items):
        assert len(items) == 1
        return [_verdict(True, items[0][0].id)]

    res = plan_hands(
        FAKE_ENGINE, deck, "test-deck", hands, [], NAMES,
        lambda prompts: [resp], fake_verify, repairs=0,
    )
    r = res[0]
    assert r["ok_line"].id == "test-deck/plan-h0-a0"  # no -c tag
    assert r["extras"] == [] and len(r["attempts"]) == 1


# --- --session plumbing ---


def test_plan_lines_session_plumbing(tmp_path, monkeypatch):
    mod = _load("plan_lines")
    seen: dict = {}

    def fake_require(session="lines"):
        seen["require"] = session
        return {"instance_id": 1}

    def fake_remote(items, timeout_per_item, jobs=None, session="lines"):
        seen["remote"] = session
        out = []
        for job in items:
            lid = job[0].get("id", "x") if isinstance(job[0], dict) else "x"
            out.append(_verdict(True, lid))
        return out

    monkeypatch.setattr(mod, "require_session", fake_require)
    monkeypatch.setattr(mod, "remote_verify", fake_remote)
    monkeypatch.setattr(mod, "Engine", lambda: FAKE_ENGINE)
    monkeypatch.setattr(mod, "load_ydk", lambda path: _deck())
    monkeypatch.setattr(
        mod._llm, "complete_many",
        lambda prompts, **kw: [_resp(_cand("only", [SAGE_N])) for _ in prompts],
    )
    out_dir = tmp_path / "planned"
    argv = ["plan_lines", "--deck", "blue-eyes", "--hands", "1", "--seed", "0",
            "--remote", "--session", "box2", "--out-dir", str(out_dir), "--jobs", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    assert seen["require"] == "box2"
    assert seen["remote"] == "box2"
    recs = (out_dir / "verified" / "blue-eyes.jsonl").read_text().splitlines()
    assert len(recs) == 1 and json.loads(recs[0])["line_id"].endswith("-a0")


def test_plan_lines_session_default_is_lines(tmp_path, monkeypatch):
    mod = sys.modules.get("plan_lines") or _load("plan_lines")
    seen: dict = {}

    def fake_require(session="lines"):
        seen["require"] = session
        return {"instance_id": 1}

    def fake_remote(items, timeout_per_item, jobs=None, session="lines"):
        seen["remote"] = session
        return [_verdict(True, job[0].get("id", "x")) for job in items]

    monkeypatch.setattr(mod, "require_session", fake_require)
    monkeypatch.setattr(mod, "remote_verify", fake_remote)
    monkeypatch.setattr(mod, "Engine", lambda: FAKE_ENGINE)
    monkeypatch.setattr(mod, "load_ydk", lambda path: _deck())
    monkeypatch.setattr(
        mod._llm, "complete_many",
        lambda prompts, **kw: [_resp(_cand("only", [SAGE_N])) for _ in prompts],
    )
    out_dir = tmp_path / "planned"
    argv = ["plan_lines", "--deck", "blue-eyes", "--hands", "1", "--seed", "0",
            "--remote", "--out-dir", str(out_dir), "--jobs", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    assert seen["require"] == "lines" and seen["remote"] == "lines"


def test_improve_lines_session_plumbing(tmp_path, monkeypatch):
    mod = _load("improve_lines")
    seen: dict = {}

    old = ComboLine(
        id="d/l0", deck="d", source_file="planner",
        hand=[SAGE_N, PWWB_N, BEWD_N, SAGE_N, PWWB_N],
        end_board=[SAGE_N], going="first",
    )
    from decklight.demos.schema import Step

    old.steps = [Step(i=0, action="normal_summon", card=SAGE_N, text="x"),
                 Step(i=1, action="activate", card=PWWB_N, text="x")]
    old.end_board_codes = [None]
    src = tmp_path / "src"
    (src / "lines").mkdir(parents=True)
    (src / "verified").mkdir(parents=True)
    (src / "lines" / "d.jsonl").write_text(json.dumps(old.to_json()) + "\n")
    (src / "verified" / "d.jsonl").write_text(json.dumps({"line_id": "d/l0", "executed_steps": 1}) + "\n")

    def fake_require(session="lines"):
        seen["require"] = session
        return {"instance_id": 1}

    def fake_remote(items, timeout_per_item, jobs=None, session="lines"):
        seen["remote"] = session
        out = []
        for job in items:
            lid = job[0].get("id", "x") if isinstance(job[0], dict) else "x"
            v = _verdict(True, lid, executed=5)
            v.end_field_codes = [SAGE, PWWB]
            out.append(v)
        return out

    new_json = json.dumps({"lines": [{
        "title": "stronger", "hand": ["PLACEHOLDER"],
        "steps": [{"i": 0, "action": "normal_summon", "card": SAGE_N, "text": "x"},
                  {"i": 1, "action": "activate", "card": PWWB_N, "text": "x"}],
        "end_board": [SAGE_N, PWWB_N], "going": "first"}]})
    monkeypatch.setattr(mod, "require_session", fake_require)
    monkeypatch.setattr(mod, "remote_verify", fake_remote)
    monkeypatch.setattr(mod, "Engine", lambda: FAKE_ENGINE)
    monkeypatch.setattr(mod, "load_ydk", lambda path: _deck())
    monkeypatch.setattr(mod._llm, "complete_many", lambda prompts, **kw: [new_json for _ in prompts])
    argv = ["improve_lines", "--deck", "d", "--src", str(src),
            "--remote", "--session", "box7", "--jobs", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    assert seen["require"] == "box7"
    assert seen["remote"] == "box7"
    kept = (src / "improved" / "verified" / "d.jsonl").read_text().splitlines()
    assert len(kept) == 1  # strictly stronger (2 vs 1) kept
