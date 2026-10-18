"""The rules glossary (web/glossary.js): shape, unique ids, short length, unique aliases.

The GLOSSARY array literal is valid JSON on purpose, so the test reads the file
as text and parses it without a JS runtime (web-ux-plan step 5, item 6).
"""

import json
import re
from pathlib import Path

GLOSSARY_JS = Path(__file__).resolve().parents[2] / "web" / "glossary.js"
_ARRAY = re.compile(r"export const GLOSSARY = (\[.*?\n\]);", re.DOTALL)


def _glossary() -> list[dict]:
    text = GLOSSARY_JS.read_text(encoding="utf-8")
    m = _ARRAY.search(text)
    assert m, "export const GLOSSARY = [...] not found in web/glossary.js"
    data = json.loads(m.group(1))
    assert isinstance(data, list)
    return data


def test_at_least_24_entries():
    assert len(_glossary()) >= 24


def test_entry_shape_and_ids_unique():
    ids = [g["id"] for g in _glossary()]
    assert len(ids) == len(set(ids)), f"duplicate ids: {sorted(ids)}"
    for g in _glossary():
        assert set(g) == {"id", "term", "aliases", "short", "long"}, g
        assert g["id"] and g["term"] and g["short"] and g["long"]
        assert isinstance(g["aliases"], list) and g["aliases"], g["id"]


def test_short_is_one_line_within_120():
    for g in _glossary():
        assert "\n" not in g["short"], g["id"]
        assert len(g["short"]) <= 120, (g["id"], len(g["short"]))


def test_aliases_unique_across_entries():
    seen: dict[str, str] = {}
    for g in _glossary():
        for a in g["aliases"]:
            key = a.lower()
            assert key not in seen, f"alias {a!r} in {g['id']} and {seen[key]}"
            seen[key] = g["id"]
