"""Card-level holdout audit (src/decklight/decks/holdout.py), all on temp dirs."""

import json
from pathlib import Path

from decklight.decks.holdout import (
    card_holdout_codes,
    check_records,
    deck_audit_rows,
    newcard_codes,
    record_card_counts,
    scenario_card_names,
    scenario_codes,
    staples,
    training_cards,
)
from decklight.decks.ydk import load_ydk

STAPLE = 90000001
A_CARD = 90000002
B_CARD = 90000003
B_EXTRA = 90000004


class StubNames:
    """Duck-typed CardNames: exact mapping only, never guesses."""

    def __init__(self, mapping: dict[str, int]) -> None:
        self.mapping = mapping
        self.names = {c: n for n, c in mapping.items()}

    def resolve(self, name: str) -> list[int]:
        return [self.mapping[name]] if name in self.mapping else []


def _write_ydk(path: Path, main: list[int], extra: list[int] | None = None) -> None:
    text = "#main\n" + "\n".join(map(str, main)) + "\n"
    if extra:
        text += "#extra\n" + "\n".join(map(str, extra)) + "\n"
    text += "!side\n"
    path.write_text(text, encoding="utf-8")


def _write_record(path: Path, main0: list[int], main1: list[int], rid: str = "r1") -> None:
    rec = {
        "line_id": rid,
        "config": {
            "deck0_main": main0,
            "deck0_extra": [],
            "deck1_main": main1,
            "deck1_extra": [],
        },
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")


def _layout(tmp_path: Path) -> dict[str, Path]:
    pool = tmp_path / "pool"
    demos = tmp_path / "demos"
    gauntlet = tmp_path / "gauntlet"
    scenarios = tmp_path / "scenarios"
    for d in (pool, demos, gauntlet, scenarios):
        d.mkdir()
    # three decks sharing one staple; deck-b is held out at deck level
    _write_ydk(pool / "deck-a.ydk", [STAPLE, A_CARD, A_CARD])
    _write_ydk(pool / "deck-b.ydk", [STAPLE, B_CARD, B_CARD], [B_EXTRA])
    _write_ydk(pool / "deck-c.ydk", [STAPLE, 90000005])
    (tmp_path / "HOLDOUT").write_text("deck-b\n", encoding="utf-8")
    return {"pool": pool, "demos": demos, "gauntlet": gauntlet, "scenarios": scenarios}


def test_staples_computed_by_deck_count(tmp_path):
    lay = _layout(tmp_path)
    decks = {p.stem: load_ydk(p) for p in sorted((lay["pool"]).glob("*.ydk"))}
    assert staples(decks, min_decks=3) == {STAPLE}
    assert staples(decks, min_decks=2) == {STAPLE}
    assert staples(decks, min_decks=4) == set()


def test_held_out_only_card_is_unseen(tmp_path):
    lay = _layout(tmp_path)
    _write_record(lay["demos"] / "r.jsonl", [STAPLE, A_CARD], [STAPLE, 90000005])
    counts = training_cards(
        [lay["demos"]],
        pool_dir=lay["pool"],
        gauntlet_dir=lay["gauntlet"],
        scenario_dir=lay["scenarios"],
        holdout_file=tmp_path / "HOLDOUT",
        names=StubNames({}),
    )
    assert counts[A_CARD] > 0  # in a record and in non-holdout deck-a
    assert counts[B_CARD] == 0  # only deck-b (held out) uses it
    assert counts[B_EXTRA] == 0


def test_check_records_flags_held_out_card(tmp_path):
    lay = _layout(tmp_path)
    (tmp_path / "CARD_HOLDOUT").write_text("deck-b\n", encoding="utf-8")
    codes = card_holdout_codes(
        pool_dir=lay["pool"], holdout_file=tmp_path / "CARD_HOLDOUT", min_decks=3
    )
    assert codes == {B_CARD, B_EXTRA}  # staple excluded
    bad = lay["demos"] / "bad.jsonl"
    _write_record(bad, [STAPLE, A_CARD], [B_CARD], rid="leaky")
    good = lay["demos"] / "good.jsonl"
    _write_record(good, [STAPLE, A_CARD], [STAPLE], rid="clean")
    violations = check_records([str(bad), str(good)], codes=codes)
    assert any(str(B_CARD) in v and "leaky" in v for v in violations), violations
    assert not any("clean" in v for v in violations), violations
    deck_hits = check_records([str(lay["pool"] / "deck-b.ydk")], codes=codes)
    assert any(str(B_CARD) in v for v in deck_hits), deck_hits


def test_card_holdout_codes_empty_without_file(tmp_path):
    assert card_holdout_codes(pool_dir=tmp_path, holdout_file=tmp_path / "MISSING") == set()
    assert check_records([str(tmp_path)], codes=set()) == []


def test_newcard_codes_on_temp_dir(tmp_path):
    lay = _layout(tmp_path)
    nc = tmp_path / "newcards"
    nc.mkdir()
    _write_ydk(nc / "rogue.ydk", [STAPLE, 90000006, 90000006], [90000007])
    codes = newcard_codes(newcards_dir=nc, pool_dir=lay["pool"], min_decks=3)
    assert codes == {90000006, 90000007}  # staple excluded
    assert newcard_codes(newcards_dir=tmp_path / "NODIR", pool_dir=lay["pool"]) == set()


def test_check_records_flags_newcards_code_by_default(tmp_path):
    lay = _layout(tmp_path)
    nc = tmp_path / "newcards"
    nc.mkdir()
    _write_ydk(nc / "rogue.ydk", [STAPLE, 90000006], [90000007])
    bad = lay["demos"] / "bad.jsonl"
    _write_record(bad, [STAPLE], [90000006], rid="newleak")
    violations = check_records(
        [str(bad)],
        pool_dir=lay["pool"],
        holdout_file=tmp_path / "MISSING",  # no CARD_HOLDOUT: only newcards codes apply
        newcards_dir=nc,
        min_decks=3,
    )
    assert any("90000006" in v and "newleak" in v for v in violations), violations


def test_scenario_names_map_and_unknowns_reported(tmp_path):
    lay = _layout(tmp_path)
    scen = lay["scenarios"] / "s1.json"
    scen.write_text(
        json.dumps(
            {
                "p0": {"hand": ["Alpha Dragon", {"card": "Beta Spell", "pos": "set"}], "mzone": []},
                "p1": {"hand": ["Zzz Not A Real Card"], "grave": []},
            }
        ),
        encoding="utf-8",
    )
    assert scenario_card_names(scen) == ["Alpha Dragon", "Beta Spell", "Zzz Not A Real Card"]
    counts, unknowns = scenario_codes(
        lay["scenarios"], names=StubNames({"Alpha Dragon": 111, "Beta Spell": 222})
    )
    assert counts[111] == 1 and counts[222] == 1
    assert any("Zzz Not A Real Card" in u for u in unknowns), unknowns
    assert all("Zzz" not in str(c) for c in counts)


def test_record_card_counts_and_audit_rows(tmp_path):
    lay = _layout(tmp_path)
    _write_record(lay["demos"] / "r1.jsonl", [STAPLE, A_CARD], [STAPLE, A_CARD], rid="r1")
    _write_record(lay["demos"] / "r2.jsonl", [STAPLE], [STAPLE], rid="r2")
    counts, n = record_card_counts([lay["demos"]])
    assert n == 2
    assert counts[A_CARD] == 1 and counts[STAPLE] == 2
    rows = deck_audit_rows(
        pool_dir=lay["pool"],
        holdout_file=tmp_path / "HOLDOUT",
        training={A_CARD: 3, 90000005: 1},
        record_counts=counts,
        min_decks=3,
    )
    by_stem = {r["stem"]: r for r in rows}
    assert by_stem["deck-b"]["holdout"] is True
    assert by_stem["deck-b"]["unseen_frac"] == 1.0  # B cards appear in no training
    assert by_stem["deck-a"]["unseen_frac"] == 0.0  # its only non-staple card is seen
    assert rows[0]["stem"] == "deck-b"  # sorted by unseen fraction first
