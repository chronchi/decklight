"""LLM combo planner: prompt building, round/repair logic, record shape."""

import json
from types import SimpleNamespace

from decklight.decks import Deck
from decklight.demos.names import CardNames
from decklight.demos.plan import (
    card_sheet,
    plan_hands,
    plan_prompt,
    repair_prompt,
    sample_hands,
)
from decklight.demos.records import make_record
from decklight.demos.schema import ComboLine, Step
from decklight.demos.verify import Verdict

NAMES = CardNames()
SAGE, PWWB, BLUE_EYES = 8240199, 63356631, 89631139
FAKE_ENGINE = SimpleNamespace(cards={})


def _deck() -> Deck:
    main = (SAGE, PWWB, BLUE_EYES, SAGE, PWWB, BLUE_EYES, 48800175, 97268402, 38517737, 89631139)
    return Deck(main, (), (), "test-deck")


def _line_json(title="sage opener") -> str:
    return json.dumps(
        {
            "lines": [
                {
                    "title": title,
                    "hand": ["PLACEHOLDER"],
                    "steps": [
                        {"i": 0, "action": "normal_summon", "card": "Sage with Eyes of Blue", "text": "x"},
                        {"i": 1, "action": "set_spelltrap", "card": "Phoenix Wing Wind Blast", "text": "y"},
                    ],
                    "end_board": ["Sage with Eyes of Blue", "Phoenix Wing Wind Blast"],
                    "going": "first",
                }
            ]
        }
    )


def _verdict(ok: bool, line_id: str, reason: str = "ok") -> Verdict:
    return Verdict(line_id, ok, reason, [0], ["stated", "auto"], 0, 3, [], f"{reason} detail", [])


def test_sample_hands_reproducible_and_no_replacement():
    d = Deck(tuple(range(100, 120)), (), (), "distinct")  # distinct codes: sample = no reuse
    h1 = sample_hands(d, 4, seed=7)
    h2 = sample_hands(d, 4, seed=7)
    assert h1 == h2 and len(h1) == 4 and all(len(h) == 5 for h in h1)
    assert all(len(set(h)) == len(h) for h in h1)  # no replacement within a hand
    assert sample_hands(d, 4, seed=8) != h1
    # Real decks hold copies: drawing the same code twice is one copy each.
    h3 = sample_hands(_deck(), 4, seed=7)
    assert all(len(h) == 5 for h in h3)


def test_card_sheet_lists_copies():
    sheet = card_sheet(FAKE_ENGINE, Deck((BLUE_EYES, BLUE_EYES, SAGE), (), (), "d"), NAMES)
    assert "Blue-Eyes White Dragon x2" in sheet
    assert "Sage with Eyes of Blue x1" in sheet
    assert sheet.count("\n") == 1  # one line per distinct card


def test_plan_prompt_states_hand_rules_and_examples():
    ex = ComboLine.from_json(json.loads(_line_json())["lines"][0])
    p = plan_prompt("test-deck", "SHEET", ["Sage with Eyes of Blue", "X"], [ex])
    assert "Sage with Eyes of Blue" in p
    assert "exactly the 5 cards" in p and "Normal Summon at most once" in p
    assert "once per turn" in p and '"lines"' in p
    assert "sage opener" in p  # example included
    assert "results`" in p  # attribution rules copied from extract
    p0 = plan_prompt("test-deck", "SHEET", ["A"], [])
    assert "EXAMPLES" not in p0


def test_repair_prompt_contains_answer_reason_and_explain():
    r = repair_prompt("BASE", '{"lines":[]}', "no_anchor@1: anchor never legal", "my hand: [...]")
    assert "BASE" in r and '{"lines":[]}' in r
    assert "no_anchor@1" in r and "my hand" in r


def test_plan_hands_repairs_failure_then_verifies():
    deck = _deck()
    hands = [[SAGE, PWWB, BLUE_EYES, SAGE, PWWB]]
    seen: list[str] = []

    def fake_complete(prompts: list[str]) -> list[str]:
        seen.extend(prompts)
        return [_line_json()]

    def fake_verify(items: list[tuple]) -> list[Verdict]:
        out = []
        for line, _d, _b in items:
            ok = line.id.endswith("-a1")  # fail round 0, verify the repair
            out.append(_verdict(ok, line.id, "ok" if ok else "no_anchor@1"))
        return out

    res = plan_hands(FAKE_ENGINE, deck, "test-deck", hands, [], NAMES, fake_complete, fake_verify, repairs=2)
    assert len(seen) == 2  # one call per round (concurrent batch)
    assert "engine rejected it" in seen[1] and "no_anchor@1" in seen[1]
    assert len(res) == 1
    r = res[0]
    assert [a["reason"] for a in r["attempts"]] == ["no_anchor@1", "ok"]
    assert r["ok_line"] is not None and r["ok_verdict"].reason == "ok"
    # Hand forcing, ids, going-first.
    assert r["ok_line"].hand == [NAMES.names[c] for c in hands[0]]
    assert r["ok_line"].going == "first" and r["ok_line"].id == "test-deck/plan-h0-a1"
    assert r["ok_line"].deck == "test-deck" and r["ok_line"].source_file == "planner"
    assert r["attempts"][0]["line"].id == "test-deck/plan-h0-a0"


def test_plan_hands_unparsable_goes_to_repair():
    deck = _deck()
    hands = [[SAGE, PWWB, BLUE_EYES, SAGE, PWWB]]
    calls = {"n": 0}

    def fake_complete(prompts: list[str]) -> list[str]:
        calls["n"] += 1
        return ["not json at all"] if calls["n"] == 1 else [_line_json()]

    def fake_verify(items: list[tuple]) -> list[Verdict]:
        return [_verdict(True, ln.id) for ln, _d, _b in items]

    res = plan_hands(FAKE_ENGINE, deck, "test-deck", hands, [], NAMES, fake_complete, fake_verify, repairs=1)
    assert [a["reason"] for a in res[0]["attempts"]] == ["unparsable", "ok"]
    assert res[0]["attempts"][0]["line"] is None
    assert res[0]["ok_line"] is not None


def test_make_record_keys_and_values():
    deck = _deck()
    line = ComboLine.from_json(json.loads(_line_json())["lines"][0])
    line.id, line.deck, line.source_file = "d/plan-h0-a0", "d", "planner"
    line.hand, line.hand_codes = ["Sage with Eyes of Blue"], [SAGE]
    for s, c in zip(line.steps, (SAGE, PWWB)):
        s.card_code = c
    line.end_board_codes = [SAGE, PWWB]
    v = _verdict(True, line.id)
    v.deck_source = "planned"
    rec = make_record(None, line, deck, v, 0, [0], "+planned")
    assert list(rec) == [
        "line_id",
        "me",
        "source_url",
        "deck",
        "deck_source",
        "config",
        "auto_single",
        "choices",
        "labels",
        "skipped_steps",
        "executed_steps",
        "inferred_frac",
        "hand_inferred",
        "end_board",
        "end_board_codes",
        "seeds_ok",
    ]
    assert rec["deck"] == "d+planned" and rec["me"] == 0 and rec["seeds_ok"] == [0]
    assert rec["choices"] == [0] and rec["config"]["seed"] == 0
    json.dumps(rec)  # serializable


def test_plan_prompt_schema_matches_extract():
    from decklight.demos import extract

    assert "ONLY the cards that THIS step itself puts into" in extract._prompt("t", "x", "d")
    p = plan_prompt("d", "sheet", ["A"], [])
    assert "ONLY the cards that THIS step itself puts into" in p


def test_step_without_results_counts_as_executed_placeholder():
    s = Step(i=0, action="other", card="", text="pass")
    assert s.action == "other" and s.card_code is None
