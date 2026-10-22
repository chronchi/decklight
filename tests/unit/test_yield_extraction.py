"""Yield-extraction fixes: 0-step drop + deterministic global nickname fallback."""

import json

from decklight.demos.extract import extract_lines
from decklight.demos.names import (
    CardNames,
    _edits_within_one,
    is_generic_nickname,
    resolve_global_unique,
)

NAMES = CardNames()


def _line(title, hand=None, steps=None):
    d = {"title": title, "hand": hand or []}
    if steps is not None:
        d["steps"] = steps
    return d


def _step(card):
    return {"i": 0, "action": "activate", "card": card, "text": f"Activate {card}."}


def _mock(lines):
    def complete(prompt, system=None, **kw):
        assert "Deck:" in prompt
        return json.dumps({"lines": lines})

    return complete


def test_zero_step_lines_dropped_ids_stable(tmp_path):
    guide = tmp_path / "g.txt"
    guide.write_text("COMBO 1: mixed\nDo things.\n")
    lines = extract_lines(
        guide,
        "t",
        {1984618},
        names=NAMES,
        complete_fn=_mock(
            [
                _line("empty", ["Nadir Servant"], []),
                _line("real", ["Nadir Servant"], [_step("Nadir Servant")]),
                _line("no-steps-key", ["Nadir Servant"]),
            ]
        ),
    )
    assert len(lines) == 1
    assert lines[0].title == "real"
    # idx is consumed by dropped lines, so the survivor keeps its stable id
    assert lines[0].id == "t/g:1"


def test_zero_step_all_dropped(tmp_path):
    guide = tmp_path / "g.txt"
    guide.write_text("COMBO 1: nothing\nDo things.\n")
    lines = extract_lines(guide, "t", None, names=NAMES, complete_fn=_mock([_line("empty", [], [])]))
    assert lines == []


def test_edits_within_one():
    assert _edits_within_one("ecclessia", "ecclesia")  # the guide's typo (1 insertion)
    assert _edits_within_one("abc", "abc")
    assert _edits_within_one("abc", "abd")
    assert not _edits_within_one("abc", "ade")
    assert not _edits_within_one("abc", "abcdx")
    assert not _edits_within_one("maximus", "maxima")


def test_global_fallback_resolves_unique():
    ferrijit = NAMES.resolve("Tri-Brigade Ferrijit the Barren Blossom")[0]
    assert ferrijit == 26847978
    assert resolve_global_unique(NAMES, "Ferrijit") == (26847978, "Tri-Brigade Ferrijit the Barren Blossom")
    ntss = NAMES.resolve("Elder Entity N'tss")[0]
    assert ntss == 80532587
    assert resolve_global_unique(NAMES, "N'tss") == (80532587, "Elder Entity N'tss")
    herald = NAMES.resolve("Herald of the Arc Light")[0]
    assert herald == 79606837
    assert resolve_global_unique(NAMES, "Herald of Arc Light") == (79606837, "Herald of the Arc Light")


def test_global_fallback_ambiguous_stays_unresolved():
    # "maximus" is a token of 3 distinct names (Dogmatika / Crusadia / Koa'ki Meiru)
    assert resolve_global_unique(NAMES, "Maximus") is None
    # "ecclessia" is 1 edit from "ecclesia", shared by 3 distinct names
    # (Dogmatika / Incredible / Ecclesia and the Dark Dragon)
    assert resolve_global_unique(NAMES, "Ecclessia") is None
    assert resolve_global_unique(NAMES, "Dragon") is None
    assert resolve_global_unique(NAMES, "") is None
    assert resolve_global_unique(NAMES, "Not A Real Card Xyz") is None


def test_global_fallback_skips_generic_words():
    assert is_generic_nickname("Mikanko")
    assert is_generic_nickname("Light")
    assert is_generic_nickname("Branded Spell")
    assert not is_generic_nickname("Ferrijit")
    assert not is_generic_nickname("Herald of Arc Light")
    assert resolve_global_unique(NAMES, "Mikanko") is None


def test_pool_and_guide_decklist_win_over_global(tmp_path):
    # Maximus is only mappable because the guide's own decklist joins the pool;
    # Ferrijit (absent from every pool) falls through to the global fallback,
    # lands in requires, and is tagged in notes. Nadir Servant (plain resolve)
    # is never touched by either layer.
    guide = tmp_path / "g.txt"
    guide.write_text(
        "URL: https://example.com/g\n"
        + "=" * 70
        + "\nMAIN DECK\n"
        + "1x Dogmatika Maximus\n"
        + "1x Nadir Servant\n"
        + "1x Ash Blossom & Joyous Spring\n"
        + "1x Blue-Eyes White Dragon\n"
        + "1x Maiden of White\n"
        + "COMBO 1: Nadir line\n"
        + "Activate Nadir Servant to search Maximus.\n"
    )

    def complete(prompt, system=None, **kw):
        if "CANDIDATES" in prompt:  # nickname batch call
            assert "Dogmatika Maximus" in prompt  # guide decklist joined the pool
            return json.dumps({"Maximus": "Dogmatika Maximus"})
        return json.dumps(
            {
                "lines": [
                    {
                        "title": "Nadir line",
                        "hand": ["Maximus", "Ferrijit", "Nadir Servant"],
                        "steps": [
                            {
                                "i": 0,
                                "action": "activate",
                                "card": "Nadir Servant",
                                "results": ["Herald of Arc Light"],
                                "text": "Activate Nadir Servant.",
                            }
                        ],
                        "going": "first",
                    }
                ]
            }
        )

    lines = extract_lines(guide, "mikanko", {1984618}, names=NAMES, complete_fn=complete)
    assert len(lines) == 1
    line = lines[0]
    assert line.hand_codes == [95679145, 26847978, 1984618]
    assert line.steps[0].results_codes == [79606837]
    # Herald appears only in the steps, so its out-of-pool code joins requires;
    # Ferrijit is already covered via the hand, so it is only tagged, not duplicated.
    assert line.requires == ["Herald of the Arc Light"]
    assert line.requires_codes == [79606837]
    assert line.unresolved_names == []
    assert "global_fallback: Ferrijit->Tri-Brigade Ferrijit the Barren Blossom" in line.notes
    assert "global_fallback: Herald of Arc Light->Herald of the Arc Light" in line.notes
    assert "Maximus->" not in line.notes
