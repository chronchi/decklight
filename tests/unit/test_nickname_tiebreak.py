"""Nickname tiebreak: ambiguous global fallback resolved via guide context."""

import json

from decklight.demos.extract import extract_lines
from decklight.demos.names import CardNames, resolve_global_unique

NAMES = CardNames()

DOGMATIKA_CTX = ["Dogmatika Punishment"]


def test_maximus_tiebreak_to_dogmatika():
    assert resolve_global_unique(NAMES, "Maximus", DOGMATIKA_CTX) == (
        95679145,
        "Dogmatika Maximus",
    )


def test_ecclessia_tiebreak_to_dogmatika():
    assert resolve_global_unique(NAMES, "Ecclessia", DOGMATIKA_CTX) == (
        60303688,
        "Dogmatika Ecclesia, the Virtuous",
    )


def test_no_context_or_tied_context_stays_unresolved():
    assert resolve_global_unique(NAMES, "Maximus") is None
    assert resolve_global_unique(NAMES, "Maximus", []) is None
    assert resolve_global_unique(NAMES, "Ecclessia") is None
    assert resolve_global_unique(NAMES, "Ecclessia", []) is None
    tied = ["Dogmatika Punishment", "Crusadia Testament"]
    assert resolve_global_unique(NAMES, "Maximus", tied) is None
    # context with no archetype overlap also leaves it unresolved
    assert resolve_global_unique(NAMES, "Maximus", ["Blue-Eyes White Dragon"]) is None


def test_unique_matches_unchanged_by_context():
    assert resolve_global_unique(NAMES, "Ferrijit") == (
        26847978,
        "Tri-Brigade Ferrijit the Barren Blossom",
    )
    assert resolve_global_unique(NAMES, "Ferrijit", DOGMATIKA_CTX) == (
        26847978,
        "Tri-Brigade Ferrijit the Barren Blossom",
    )
    assert resolve_global_unique(NAMES, "Ferrijit", ["Crusadia Testament"]) == (
        26847978,
        "Tri-Brigade Ferrijit the Barren Blossom",
    )


def test_extract_wires_guide_context_into_fallback(tmp_path):
    # Pool holds Dogmatika Punishment (archetype context) but not Maximus; the
    # LLM stub leaves Maximus unresolved so the global fallback must resolve
    # it to Dogmatika Maximus via the pool context.
    guide = tmp_path / "g.txt"
    guide.write_text("COMBO 1: dogmatika line\nSearch Maximus with Nadir Servant.\n")

    def complete(prompt, system=None, **kw):
        if "CANDIDATES" in prompt:  # nickname batch call: resolve nothing
            return json.dumps({"Maximus": None})
        return json.dumps(
            {
                "lines": [
                    {
                        "title": "dogmatika line",
                        "hand": ["Maximus", "Nadir Servant"],
                        "steps": [
                            {
                                "i": 0,
                                "action": "activate",
                                "card": "Nadir Servant",
                                "text": "Activate Nadir Servant.",
                            }
                        ],
                        "going": "first",
                    }
                ]
            }
        )

    lines = extract_lines(guide, "mikanko", {1984618, 82956214}, names=NAMES, complete_fn=complete)
    assert len(lines) == 1
    line = lines[0]
    assert line.hand_codes == [95679145, 1984618]
    assert line.unresolved_names == []
    assert "global_fallback: Maximus->Dogmatika Maximus" in line.notes
