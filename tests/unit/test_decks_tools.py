"""Deck audit (check_decks) and deck sheet tools."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


check_decks = _load("check_decks")
deck_sheet = _load("deck_sheet")


def test_audit_flags_synthetic_deck_missing_namesake(tmp_path):
    """Three deck cards quoting "Dark Magician" without the card itself -> flagged."""
    from decklight.demos.names import CardNames

    p = tmp_path / "syn.ydk"
    p.write_text("#main\n12266229\n47222536\n48680970\n#extra\n37818794\n!side\n")
    rows = check_decks.audit(p, CardNames())
    assert any(name == "Dark Magician" for name, _, _ in rows), rows


def test_audit_fixed_dark_magician_has_its_namesake():
    from decklight.demos.names import CardNames

    rows = check_decks.audit(ROOT / "decks" / "pool" / "dark-magician.ydk", CardNames())
    assert all(name != "Dark Magician" for name, _, _ in rows), rows


def test_sheet_lists_each_card_once():
    text = deck_sheet.sheet("snake-eye")
    assert "== Main" in text and "== Extra" in text
    ash = [line for line in text.splitlines() if line.startswith("Snake-Eye Ash ")]
    assert len(ash) == 1 and "[FIRE Pyro/" in ash[0] and " — " in ash[0]
