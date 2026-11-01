"""deck_coverage.json (scripts/build_deck_coverage.py) is current and names real lobby decks."""

import json
from pathlib import Path

from decklight.web import server

ROOT = Path(__file__).resolve().parents[2]


def test_committed_coverage_matches_a_rebuild():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bdc", ROOT / "scripts" / "build_deck_coverage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert json.loads(server.COVERAGE_PATH.read_text()) == json.loads(json.dumps(mod.build()))


def test_coverage_names_lobby_decks_with_both_levels():
    cov = server._load_coverage()
    lobby = {p.stem for _g, d in server.DECK_DIRS for p in d.glob("*.ydk")}
    assert cov and set(cov) <= lobby
    assert {v["level"] for v in cov.values()} == {"trained", "related"}
    assert all(v["level"] == "trained" or v["overlap"] >= 0.5 for v in cov.values())
    assert "adamancipator" not in cov  # an anchor deck: no specialist (user's question, 2026-10-31)
