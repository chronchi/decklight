"""deck_index.json (scripts/build_deck_index.py) is current and covers every lobby deck."""

import importlib.util
import json
from pathlib import Path

from decklight.web import server

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "src" / "decklight" / "web" / "deck_index.json"


def test_committed_index_matches_a_rebuild():
    spec = importlib.util.spec_from_file_location("bdi", ROOT / "scripts" / "build_deck_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert json.loads(INDEX.read_text()) == json.loads(json.dumps(mod.build()))


def test_index_covers_the_lobby():
    idx = json.loads(INDEX.read_text())
    lobby = {p.stem for _g, d in server.DECK_DIRS for p in d.glob("*.ydk")}
    assert set(idx["decks"]) == lobby
    for v in idx["decks"].values():
        assert len(v["sig"]) == 3 and v["main"] and all(str(c) in idx["names"] for c, _n in v["main"] + v["extra"])
