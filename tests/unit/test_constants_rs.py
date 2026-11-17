"""The checked-in Rust constants must be exactly what `scripts/app/gen_constants_rs.py` renders (A1b)."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gen_constants_rs", ROOT / "scripts" / "app" / "gen_constants_rs.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_constants_rs"] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load()


def test_constants_rs_is_current():
    want = G.render(G.constants())
    got = G.RS_OUT.read_text(encoding="utf-8")
    assert got == want, "rust/decklight-core/src/constants.rs is stale: run scripts/app/gen_constants_rs.py"


def test_every_python_constant_is_rendered_with_its_value():
    items = G.constants()
    assert len(items) > 300
    text = G.render(items)
    for name, value in items:
        assert f"pub const {name}:" in text
        assert f"0x{value & 0xFFFF_FFFF_FFFF_FFFF:X}" in text or f"= {value};" in text
