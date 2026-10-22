"""Planner end-to-end with the real engine: a fake LLM's good line verifies."""

import json

import pytest

from decklight.demos.dataset import iter_samples
from decklight.demos.explain import explain_failure
from decklight.demos.names import CardNames
from decklight.demos.plan import plan_hands
from decklight.demos.records import make_record
from decklight.demos.verify import verify_line

cpp_env = pytest.importorskip("decklight.engine.cpp_env", reason="extension not built")

SAGE = 8240199
PWWB = 63356631


def _good_line_json(names: CardNames) -> str:
    sage, pwwb = names.names[SAGE], names.names[PWWB]
    return json.dumps(
        {
            "lines": [
                {
                    "title": "sage plus trap",
                    "hand": ["whatever"],
                    "steps": [
                        {"i": 0, "action": "normal_summon", "card": sage, "text": f"Normal Summon {sage}."},
                        {"i": 1, "action": "set_spelltrap", "card": pwwb, "text": f"Set {pwwb}."},
                    ],
                    "end_board": [sage, pwwb],
                    "end_board_text": f"{sage} plus set {pwwb}",
                    "going": "first",
                }
            ]
        }
    )


def test_planned_good_line_verifies_and_replays(engine, decks, tmp_path):
    deck = decks["BlueEyes"]
    names = CardNames()
    hand = [SAGE, PWWB, 89631139, 48800175, 97268402]
    assert all(c in deck.main for c in hand)

    def fake_complete(prompts: list[str]) -> list[str]:
        assert len(prompts) == 1 and names.names[SAGE] in prompts[0]
        return [_good_line_json(names)]

    def real_verify(items: list[tuple]) -> list:
        return [verify_line(engine, ln, d, seed=0, budget=b) for ln, d, b in items]

    res = plan_hands(engine, deck, "blue-eyes", [hand], [], names, fake_complete, real_verify)
    assert res[0]["ok_verdict"] is not None, res[0]["attempts"][0]["detail"]
    line, verdict = res[0]["ok_line"], res[0]["ok_verdict"]
    assert verdict.ok and line.hand == [names.names[c] for c in hand]

    verdict.deck_source = "planned"
    rec = make_record(engine, line, deck, verdict, 0, [0], "+planned")
    assert rec["deck"] == "blue-eyes+planned" and rec["me"] == 0
    f = tmp_path / "planned.jsonl"
    f.write_text(json.dumps(rec) + "\n")
    samples = list(iter_samples(engine, f))
    assert len(samples) > 0  # record replays into BC samples
    assert all(w > 0 for _, _, w, _ in samples)


def test_explain_failure_reports_deepest_point(engine, decks):
    from decklight.demos.schema import ComboLine, Step

    deck = decks["BlueEyes"]
    names = CardNames()
    line = ComboLine(
        id="t",
        deck="t",
        hand=[str(SAGE), str(PWWB)],
        hand_codes=[SAGE, PWWB],
        steps=[
            Step(i=0, action="normal_summon", card=str(SAGE), card_code=SAGE, text="ns"),
            Step(i=1, action="set_spelltrap", card=str(PWWB), card_code=PWWB, text="set"),
        ],
        end_board=["Blue-Eyes White Dragon"],
        end_board_codes=[89631139],
    )
    v = verify_line(engine, line, deck, seed=0, budget=500)
    assert not v.ok
    text = explain_failure(engine, line, deck, v, names.names)
    assert text.startswith(f"{v.reason}: ")
    assert "step 0: normal_summon" in text and "step 1: set_spelltrap" in text
    assert ("options:" in text) or ("deepest path ends the duel" in text)
