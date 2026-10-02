"""Stage-3 verifier: a ComboLine executes from a puzzle start and reaches its end board."""

import pytest

from ygo_zero.demos.schema import ComboLine, Step
from ygo_zero.demos.verify import make_config, verify_line
from ygo_zero.engine.cpp_env import CppEnv

cpp_env = pytest.importorskip("ygo_zero.engine.cpp_env", reason="extension not built")

SAGE = 8240199  # Sage with Eyes of Blue: level 1, normal-summonable
BLUE_EYES = 89631139


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


def test_normal_summon_line_verifies_and_replays(engine, decks):
    deck = decks["BlueEyes"]
    line = _line([SAGE], [_step(0, "normal_summon", SAGE)], [SAGE])
    v = verify_line(engine, line, deck, seed=0)
    assert v.ok, v.detail
    assert v.reason == "ok" and v.labels[0] == "stated"
    assert v.skipped_steps == [] and v.deck_source == ""
    assert _my_field(engine, make_config(line, deck, 0), v.choices) == sorted(v.end_field_codes)


def test_impossible_end_board_is_rejected(engine, decks):
    deck = decks["BlueEyes"]
    line = _line([SAGE], [_step(0, "normal_summon", SAGE)], [BLUE_EYES])
    v = verify_line(engine, line, deck, seed=0, budget=500)
    assert not v.ok and v.reason in ("end_board_mismatch", "budget"), (v.reason, v.detail)


def test_card_not_in_deck_is_missing(engine, decks):
    deck = decks["BlueEyes"]
    line = _line([SAGE], [_step(0, "normal_summon", 12345678, results=[12345678])], [12345678])
    v = verify_line(engine, line, deck, seed=0)
    assert not v.ok and v.reason == "missing_card", (v.reason, v.detail)
