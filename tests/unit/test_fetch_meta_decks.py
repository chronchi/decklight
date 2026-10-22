"""fetch_meta_decks: pure helpers on a small fake page (no network, no engine lib)."""

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from decklight.decks import parse_ydk

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fm = _load("fetch_meta_decks")

TODAY = date(2026, 10, 3)

DESC = (
    "<p>Category: Tournament Meta Decks (TCG)</p>"
    "<p>Tournament: Mexico City WCQ Regional &ndash; September 27th 2026</p>"
    "<p>Placement: Top 8</p>"
)


def _entry(
    deck_num=7, name="Elfnote", desc=DESC, n_main=40, n_extra=15, n_side=15, pretty="elfnote-7", fields=True
):
    base = 1000 + (deck_num or 0) * 1000
    e = {
        "deck_name": name,
        "deck_description": desc,
        "main_deck": json.dumps([str(base + i) for i in range(n_main)]),
        "extra_deck": json.dumps([str(base + 500 + i) for i in range(n_extra)]),
        "side_deck": json.dumps([str(base + 900 + i) for i in range(n_side)]),
        "deckNum": deck_num,
        "pretty_url": pretty,
        "submit_date": "6 days ago",
    }
    if fields:
        e["tournamentName"] = "Mexico City WCQ Regional"
        e["tournamentPlacement"] = "Top 8"
    return e


def _fake_db(codes, extra=()):
    return SimpleNamespace(cards={c: SimpleNamespace(type=fm.EXTRA_T if c in extra else 0) for c in codes})


def _fake_names():
    return SimpleNamespace(alias={}, names={})


def test_slugify():
    assert fm.slugify("Elfnote") == "elfnote"
    assert fm.slugify("  Snake-Eye Fire!! Combo ") == "snake-eye-fire-combo"
    assert fm.slugify("Dracotail Lukias / Mululu") == "dracotail-lukias-mululu"
    assert fm.slugify("---") == ""


def test_parse_passcodes():
    assert fm.parse_passcodes('["13597785", "102380"]') == [13597785, 102380]
    assert fm.parse_passcodes('["12", "abc"]') is None
    assert fm.parse_passcodes("not json") is None
    assert fm.parse_passcodes('{"a": 1}') is None
    assert fm.parse_passcodes([12, 34]) == [12, 34]
    assert fm.parse_passcodes('["0"]') is None


def test_normalise_from_description():
    rec = fm.normalise(_entry(fields=False), TODAY)
    assert rec is not None
    assert rec["slug"] == "elfnote"
    assert rec["tournament"] == "Mexico City WCQ Regional"
    assert rec["placement"] == "Top 8"
    assert rec["date"] == "2026-09-27"
    assert rec["url"] == "https://ygoprodeck.com/deck/elfnote-7"
    assert len(rec["main"]) == 40 and len(rec["extra"]) == 15 and len(rec["side"]) == 15


def test_normalise_prefers_fields_and_relative_date():
    e = _entry(desc="<p>no tournament here</p>", fields=True)
    rec = fm.normalise(e, TODAY)
    assert rec is not None
    assert rec["tournament"] == "Mexico City WCQ Regional"  # field wins over empty parse
    assert rec["date"] == "2026-09-27"  # "6 days ago" fallback
    bad = _entry()
    bad["main_deck"] = '["12", "xx"]'
    assert fm.normalise(bad, TODAY) is None
    bad_name = _entry(name="---")
    assert fm.normalise(bad_name, TODAY) is None


def test_placement_rank_order():
    ranks = [
        fm.placement_rank(p)
        for p in ["1st Place", "Winner", "2nd Place", "Top 4", "Top 8", "Top 32", "unknown", ""]
    ]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1 and ranks[-1] == 9999


def test_select_per_archetype_prefers_placement_then_newest():
    def rec(num, placement, day):
        return {"slug": "elfnote", "deck_num": num, "placement": placement, "date": day}

    cands = [
        rec(1, "Top 8", "2026-09-20"),
        rec(2, "1st Place", "2026-09-01"),  # better placement, older
        rec(3, "Top 8", "2026-09-27"),  # same rank, newer
        {"slug": "invoked", "deck_num": 4, "placement": "Top 4", "date": "2026-09-27"},
    ]
    picked = fm.select(cands, per_archetype=2)
    elfnote = [c["deck_num"] for c in picked if c["slug"] == "elfnote"]
    assert elfnote == [2, 3]  # 1st Place first, then the newer Top 8
    assert [c["deck_num"] for c in picked if c["slug"] == "invoked"] == [4]
    assert fm.select(cands, per_archetype=0) == []


def test_is_holdout_deterministic_and_consistent():
    assert fm.is_holdout("brand-new-slug-xyz", set(), set(), 0.0) is False
    assert fm.is_holdout("brand-new-slug-xyz", set(), set(), 1.0) is True
    a = fm.is_holdout("some-slug", set(), set(), 0.1)
    assert fm.is_holdout("some-slug", set(), set(), 0.1) == a  # deterministic
    # existing pool membership wins over the dice
    assert fm.is_holdout("sky-striker", {"sky-striker"}, set(), 1.0) is False
    assert fm.is_holdout("blackwing", {"blackwing"}, {"blackwing"}, 0.0) is True
    assert fm.is_holdout("blackwing", set(), {"blackwing"}, 0.0) is True


def test_assign_stems_skips_existing(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "elfnote-meta1.ydk").write_text("#taken\n")
    picked = [
        {"slug": "elfnote", "name": "Elfnote"},
        {"slug": "elfnote", "name": "Elfnote"},
    ]
    rows, skipped = fm.assign_stems(picked, pool, {"elfnote"}, set(), 0.0)
    assert [s for _, s, _ in rows] == ["elfnote-meta2", "elfnote-meta3"]
    assert [sp for _, _, sp in rows] == ["train", "train"]
    assert skipped == 2  # both picks scan past the pre-existing meta1 file


def test_validate_record_monkeypatched(monkeypatch):
    rec = fm.normalise(_entry(), TODAY)
    assert rec is not None
    extra_code = rec["main"].pop()  # an extra-deck monster sitting in main
    engine = _fake_db(
        [*rec["main"], *rec["extra"], extra_code, *rec["side"]], extra=[*rec["extra"], extra_code]
    )
    rec["main"].append(extra_code)
    names = _fake_names()
    monkeypatch.setattr(fm, "validate_codes", lambda *a, **k: [])
    monkeypatch.setattr(fm, "smoke", lambda *a, **k: None)
    ok, reason = fm.validate_record(rec, engine, names)
    assert (ok, reason) == (True, "")
    assert extra_code in rec["extra"] and extra_code not in rec["main"]  # resplit by type

    monkeypatch.setattr(fm, "smoke", lambda *a, **k: "boom")
    ok, reason = fm.validate_record(dict(rec), engine, names)
    assert (ok, reason) == (False, "smoke_fail")

    monkeypatch.setattr(fm, "validate_codes", lambda *a, **k: ["X: 4 copies (max 3)"])
    monkeypatch.setattr(fm, "smoke", lambda *a, **k: None)
    ok, reason = fm.validate_record(dict(rec), engine, names)
    assert (ok, reason) == (False, "copy_limit")


def test_check_side():
    engine = _fake_db([1, 2, 3])
    assert fm.check_side(list(range(16)), engine, _fake_names()) is not None  # >15
    assert fm.check_side([999], engine, _fake_names()) is not None  # unknown code
    assert fm.check_side([1, 1, 1, 1], engine, _fake_names()) is not None  # >3 copies
    assert fm.check_side([1, 1, 1, 2], engine, _fake_names()) is None


def test_shared_validate_codes_rejects():
    from build_decks import validate_codes

    engine = _fake_db([14558127])  # Ash Blossom: has an official script
    names = _fake_names()
    probs = validate_codes([14558127] * 41, [14558127], engine, names, min_extra=0)
    assert any("copies (max 3)" in p for p in probs)
    probs = validate_codes([123456789], [], engine, names, min_extra=0)
    assert any("missing from the engine card DB" in p for p in probs)
    probs = validate_codes([14558127] * 10, [], engine, names, min_extra=0)
    assert any("need 40-60" in p for p in probs)


def test_write_ydk_roundtrip(tmp_path):
    rec = fm.normalise(_entry(), TODAY)
    assert rec is not None
    p = tmp_path / "elfnote-meta1.ydk"
    fm.write_ydk(p, rec, rec["main"], rec["extra"], "2026-10-03")
    text = p.read_text()
    assert "#source https://ygoprodeck.com/deck/elfnote-7" in text
    assert "Mexico City WCQ Regional" in text and "Top 8" in text
    deck = parse_ydk(text)
    assert list(deck.main) == rec["main"]
    assert list(deck.extra) == rec["extra"]
    assert list(deck.side) == rec["side"]


def test_assign_stems_same_slug_same_split(tmp_path):
    # a newly held-out archetype must hold out ALL its files, not just the first
    pool = tmp_path / "pool"
    pool.mkdir()
    picked = [{"slug": "doomz", "name": "DoomZ"}, {"slug": "doomz", "name": "DoomZ"}]
    rows, _ = fm.assign_stems(picked, pool, set(), set(), 1.0)
    assert [sp for _, _, sp in rows] == ["holdout", "holdout"]
    rows, _ = fm.assign_stems(picked, pool, set(), set(), 0.0)
    assert [sp for _, _, sp in rows] == ["train", "train"]
    # mid-frac: every file of one slug shares one split
    for frac in (0.1, 0.5, 0.9):
        rows, _ = fm.assign_stems(
            [{"slug": "mid-slug", "name": "X"} for _ in range(2)], pool, set(), set(), frac
        )
        assert len({sp for _, _, sp in rows}) == 1


def test_is_holdout_matches_meta_suffixed_stems():
    # an earlier fetch's <slug>-meta<k>.ydk anchors the archetype's split
    assert fm.is_holdout("elfnote-mitsurugi", {"elfnote-meta1"}, {"elfnote-meta1"}, 0.0) is True
    assert fm.is_holdout("elfnote", {"elfnote-meta1"}, set(), 1.0) is False


def test_fetch_pages_reuses_cache_with_true_offsets(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "CACHE", tmp_path)
    decks = [{"deckNum": 1}, {"deckNum": 2}]
    (tmp_path / "page-0.json").write_text(json.dumps([decks[0]]))
    (tmp_path / "page-1.json").write_text(json.dumps([decks[1]]))
    entries, fetched = fm.fetch_pages(pages=2, per_page=1, fmt="X")
    assert entries == decks and fetched == 0


def test_fetch_pages_continues_past_short_cached_first_page(tmp_path, monkeypatch):
    # the API fixes its own page size (ignores `limit`): a cached first page shorter
    # than --per-page must not end pagination.
    monkeypatch.setattr(fm, "CACHE", tmp_path)
    (tmp_path / "page-0.json").write_text(json.dumps([{"deckNum": 1}, {"deckNum": 2}]))

    class Resp:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=60):
        if "offset=2" in req.full_url:
            return Resp(json.dumps([{"deckNum": 3}, {"deckNum": 4}]).encode())
        return Resp(b"[]")

    monkeypatch.setattr(fm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fm.time, "sleep", lambda s: None)
    entries, fetched = fm.fetch_pages(pages=3, per_page=50, fmt="X")
    assert [d["deckNum"] for d in entries] == [1, 2, 3, 4]
    assert fetched == 2


def test_fake_page_end_to_end_selection():
    recs = [r for r in (fm.normalise(_entry(deck_num=i), TODAY) for i in (1, 2)) if r]
    assert len(recs) == 2
    picked = fm.select(recs, per_archetype=2)
    assert len(picked) == 2
