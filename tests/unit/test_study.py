"""Study-on-demand worker tests (plays 4 raw games, ~15 s; no scoring)."""

import json

import pytest

from decklight.engine.resources import CORE_LIB


def _swapped_ydk(src_text: str, subs: list[tuple[int, int]]) -> str:
    """The deck with main cards at `subs` indexes replaced (same size, stays legal)."""
    lines = src_text.splitlines()
    main_idx = next(i for i, l in enumerate(lines) if l.strip() == "#main")
    extra_idx = next(i for i, l in enumerate(lines) if l.strip() == "#extra")
    body = [l for l in lines[main_idx + 1 : extra_idx] if l.strip().isdigit()]
    for i, code in subs:
        body[i] = str(code)
    return (
        "#main\n"
        + "\n".join(body)
        + "\n#extra\n"
        + "\n".join(l for l in lines[extra_idx + 1 :] if l.strip().isdigit())
    )


def _run_study(monkeypatch, root, ydk_text: str, tag: str) -> tuple:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from study_deck import main

    ydk = root / f"{tag}.ydk"
    ydk.write_text(ydk_text)
    out_dir = root / "data" / "primers" / "studied"
    rc = main(
        [
            "--ydk",
            str(ydk),
            "--out-dir",
            str(out_dir),
            "--procs",
            "1",
            "--games",
            "4",
            "--no-score",
        ]
    )
    return rc, out_dir


def test_study_deck_no_score(tmp_path, monkeypatch):
    from decklight.analysis import events
    from decklight.analysis.primer import hash_of
    from decklight.decks import load_ydk, parse_ydk

    if not CORE_LIB.exists():
        pytest.skip("core not built: run scripts/build_core.sh")
    pool = events.ROOT / "decks" / "pool"
    src = (pool / "branded.ydk").read_text()
    other = parse_ydk((pool / "blue-eyes.ydk").read_text())
    ydk1 = _swapped_ydk(src, [(0, other.main[0])])
    rc, out_dir = _run_study(monkeypatch, tmp_path, ydk1, "one-card")
    assert rc == 0
    studied = json.loads((out_dir / "primers.json").read_text())
    assert len(studied["decks"]) == 1
    h1, entry = next(iter(studied["decks"].items()))
    assert entry["study"] == {"phase": "games", "games": 4}
    assert entry["games"] == 4
    assert h1 == hash_of(load_ydk(out_dir / "decks" / f"study-{h1[:10]}.ydk"))

    # The studied deck name resolves via deck_path (as if under the real root).
    with monkeypatch.context() as m:
        m.setattr(events, "ROOT", tmp_path)
        assert events.deck_path(f"study-{h1[:10]}").is_file()

    # A second run on another deck keeps the first entry.
    ydk2 = _swapped_ydk(src, [(0, other.main[0]), (1, other.main[1])])
    rc2, _ = _run_study(monkeypatch, tmp_path, ydk2, "one-card-2")
    assert rc2 == 0
    studied2 = json.loads((out_dir / "primers.json").read_text())
    h2 = hash_of(parse_ydk(ydk2, "x"))
    assert h2 != h1
    assert set(studied2["decks"]) == {h1, h2}
    for h in (h1, h2):
        assert studied2["decks"][h]["study"] == {"phase": "games", "games": 4}
