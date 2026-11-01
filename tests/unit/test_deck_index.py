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


def test_runtime_decks_get_a_card_list():
    """Fine-tunes' own and pasted decks are not in the committed index; the server adds them (user 2026-11-01)."""
    from decklight.web.deck_index import DeckIndex

    idx = DeckIndex()
    base = json.loads(INDEX.read_text())
    some = base["decks"]["snake-eye"]
    main = [c for c, n in some["main"] for _ in range(n)]
    extra = [c for c, n in some["extra"] for _ in range(n)]
    out = idx.json_for([("tuned-x", "My tune", "The deck a fine-tuned model was trained for", main, extra)])
    e = out["decks"]["tuned-x"]
    assert sorted(e["main"]) == sorted(some["main"]) and e["counts"] == some["counts"] and e["mech"] == some["mech"]
    assert len(e["sig"]) == 3 and all(str(c) in out["names"] for c, _n in e["main"] + e["extra"])
    assert "tuned-x" not in idx.json_for([])["decks"]  # the base index is not mutated
