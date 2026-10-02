"""Stage-3 verifier: a ComboLine executes from a puzzle start and reaches its end board."""

import pytest

from ygo_zero.demos.schema import ComboLine, Step
from ygo_zero.demos.verify import (
    effective_hand_codes,
    infer_hand_codes,
    quality_guard,
    verify_line,
)
from ygo_zero.engine.cpp_env import CppEnv

cpp_env = pytest.importorskip("ygo_zero.engine.cpp_env", reason="extension not built")

SAGE = 8240199  # Sage with Eyes of Blue: level 1, normal-summonable
BLUE_EYES = 89631139
MELODY = 48800175  # The Melody of Awakening Dragon: discard 1, search big dragons
PWWB = 63356631  # Phoenix Wing Wind Blast: settable trap
ALTERNATIVE = 38517737  # Blue-Eyes Alternative White Dragon
VEILER = 97268402  # Effect Veiler: level 1 Light Tuner (a Sage-trigger hit)


def _step(i: int, action: str, code: int, results: list[int] | None = None) -> Step:
    results = [code] if results is None else results
    name = str(code)
    return Step(
        i=i,
        action=action,
        card=name,
        card_code=code,
        materials=[],
        targets=[],
        results=[str(c) for c in results],
        materials_codes=[],
        targets_codes=[],
        results_codes=list(results),
        text="",
    )


def _line(hand: list[int], steps: list[Step], end: list[int], **kw) -> ComboLine:
    return ComboLine(
        id="test",
        deck="test",
        hand=[str(c) for c in hand],
        hand_codes=list(hand),
        steps=steps,
        end_board=[str(c) for c in end],
        end_board_codes=list(end),
        **kw,
    )


def _my_field(engine, config, choices) -> list[int]:
    env = CppEnv(engine, config)
    obs = env.reset()
    for i in choices:
        obs = env.step(i)
    assert obs is not None
    rows = obs["cards"][obs["card_mask"]]
    mine = rows[(rows[:, 3] ^ env.player) == 0]
    return sorted(mine[(mine[:, 1] == 3) | (mine[:, 1] == 4)][:, 0].tolist())


def test_two_step_line_verifies_and_replays(engine, decks):
    deck = decks["BlueEyes"]
    line = _line(
        [SAGE, PWWB],
        [_step(0, "normal_summon", SAGE), _step(1, "set_spelltrap", PWWB)],
        [SAGE, PWWB],
    )
    v = verify_line(engine, line, deck, seed=0)
    assert v.ok, v.detail
    assert v.reason == "ok" and v.labels[0] == "stated"
    assert v.skipped_steps == [] and v.deck_source == ""
    assert v.executed_steps == 2 and 0.0 <= v.inferred_frac <= 1.0
    assert not v.hand_inferred
    assert _my_field(engine, v.config, v.choices) == sorted(v.end_field_codes)


def test_impossible_end_board_is_rejected(engine, decks):
    deck = decks["BlueEyes"]
    line = _line(
        [SAGE, PWWB],
        [_step(0, "normal_summon", SAGE), _step(1, "set_spelltrap", PWWB)],
        [BLUE_EYES],
    )
    v = verify_line(engine, line, deck, seed=0, budget=500)
    assert not v.ok and v.reason in ("end_board_mismatch", "budget"), (v.reason, v.detail)


def test_card_not_in_deck_is_missing(engine, decks):
    deck = decks["BlueEyes"]
    line = _line([SAGE], [_step(0, "normal_summon", 12345678, results=[12345678])], [12345678])
    v = verify_line(engine, line, deck, seed=0)
    assert not v.ok and v.reason == "missing_card", (v.reason, v.detail)


def test_quality_guard_priority():
    assert quality_guard(5, [], ["stated", "inferred"]) is None
    assert quality_guard(2, [], ["stated", "stated"]) is None
    # vacuous beats everything: all skipped, or no agent choices
    assert quality_guard(1, [0], []) == "vacuous"
    assert quality_guard(3, [0, 1, 2], ["stated"]) == "vacuous"
    assert quality_guard(2, [], []) == "vacuous"
    assert quality_guard(0, [], []) == "vacuous"
    # over-skipped but not vacuous
    assert quality_guard(4, [0, 1], ["stated"] * 5) == "too_skipped"
    assert quality_guard(3, [0], ["stated"] * 5) is None  # exactly 1/3 is fine
    # fragments: <2 executed steps
    assert quality_guard(1, [], ["stated"]) == "trivial"
    assert quality_guard(3, [0, 1], ["stated"]) == "too_skipped"  # too_skipped wins over trivial


def test_single_step_execution_is_trivial(engine, decks):
    deck = decks["BlueEyes"]
    line = _line([SAGE], [_step(0, "normal_summon", SAGE)], [SAGE])
    v = verify_line(engine, line, deck, seed=0)
    assert not v.ok and v.reason == "trivial", (v.reason, v.detail)
    assert v.executed_steps == 1


def test_all_skipped_is_vacuous(engine, decks):
    deck = decks["BlueEyes"]
    # Alternative's destruction vs an empty board: anchor never legal, no results -> skipped.
    line = _line([SAGE], [_step(0, "activate", ALTERNATIVE, results=[])], [])
    v = verify_line(engine, line, deck, seed=0)
    assert not v.ok and v.reason == "vacuous", (v.reason, v.detail)
    assert v.skipped_steps == [0]


def test_mostly_skipped_is_too_skipped(engine, decks):
    deck = decks["BlueEyes"]
    # Blue-Eyes White Dragon is vanilla: ACTIVATE is never legal, so both middle steps skip.
    line = _line(
        [SAGE, PWWB],
        [
            _step(0, "normal_summon", SAGE),
            _step(1, "activate", BLUE_EYES, results=[]),
            _step(2, "activate", BLUE_EYES, results=[]),
            _step(3, "set_spelltrap", PWWB),
        ],
        [SAGE, PWWB],
    )
    v = verify_line(engine, line, deck, seed=0)
    assert not v.ok and v.reason == "too_skipped", (v.reason, v.detail)
    assert v.skipped_steps == [1, 2]
    assert v.executed_steps == 2


def test_debundled_results_carry_to_next_step(engine, decks):
    """A step whose results arrive one step late verifies via the carry (and only via it)."""
    deck = decks["BlueEyes"]
    # Step 0 bundles Blue-Eyes (which only Melody fetches in step 1); Sage's own
    # summon trigger cannot produce it (not a Level 1 Light Tuner).
    line = _line(
        [SAGE, MELODY],
        [
            _step(0, "normal_summon", SAGE, results=[SAGE, BLUE_EYES]),
            _step(1, "activate", MELODY, results=[BLUE_EYES]),
        ],
        [],
    )
    v = verify_line(engine, line, deck, seed=0)
    assert v.ok, v.detail
    assert v.executed_steps == 2
    assert "carried=[(0," in v.detail and str(BLUE_EYES) in v.detail


def test_empty_hand_is_inferred_from_leading_anchors(engine, decks):
    deck = decks["BlueEyes"]
    steps = [_step(0, "normal_summon", SAGE), _step(1, "set_spelltrap", PWWB)]
    assert infer_hand_codes(_line([], steps, [SAGE, PWWB])) == [SAGE, PWWB]
    line = _line([], steps, [SAGE, PWWB])
    v = verify_line(engine, line, deck, seed=0)
    assert v.ok, v.detail
    assert v.hand_inferred and "hand_inferred" in line.notes
    assert v.executed_steps == 2
    codes, flag = effective_hand_codes(line)
    assert flag and codes == [SAGE, PWWB]


def test_inference_stops_at_search_and_ignores_fusion():
    # Normal summon that searches: keep the summoner, not the searched card, then stop.
    line = _line(
        [],
        [
            _step(0, "normal_summon", SAGE, results=[VEILER]),
            _step(1, "set_spelltrap", PWWB),
        ],
        [],
    )
    assert infer_hand_codes(line) == [SAGE]
    # A line opening with a fusion infers nothing (its card is an extra-deck monster).
    fused = _line([], [_step(0, "fusion_summon", BLUE_EYES), _step(1, "activate", BLUE_EYES)], [])
    assert infer_hand_codes(fused) == []
    assert effective_hand_codes(fused) == ([], False)


def test_summon_step_needs_its_result_on_the_field(engine, decks):
    """Regression (Salamangreat Foxy, 2026-10-02): a summon step whose result is merely out of the deck (here a
    spell that went to the GY after activating) must not count as executed."""
    deck = decks["BlueEyes"]
    line = _line(
        [SAGE, MELODY, BLUE_EYES],
        [_step(0, "normal_summon", SAGE), _step(1, "special_summon", MELODY)],
        [SAGE],
    )
    v = verify_line(engine, line, deck, seed=0, budget=300)
    assert not v.ok, v.detail


def test_going_second_line_faces_a_board_and_attacks(engine, decks):
    """Going second: the opponent starts with a board and passes; I (player 1) summon and attack on turn 2."""
    deck = decks["BlueEyes"]
    line = _line(
        [SAGE],
        [_step(0, "normal_summon", SAGE), _step(1, "attack", SAGE, results=[])],
        [],
        going="second",
    )
    v = verify_line(engine, line, deck, seed=0, budget=500)
    assert v.ok, v.detail
    assert any(p.team == 0 and p.location != 2 for p in v.config.setup)  # opponent board placed
    assert all(p.team == 1 for p in v.config.setup if p.location == 2)  # my hand is player 1's
