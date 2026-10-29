"""forge_lists: pure split / opponent-filter / exclusion helpers (no network, no engine)."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("forge_lists", ROOT / "scripts" / "forge_lists.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["forge_lists"] = mod
    spec.loader.exec_module(mod)
    return mod


fl = _load()


# --- split_lists: newest held out, the next newest train ---


def test_split_lists_newest_heldout_rest_train():
    cands = [("a-meta1", "2026-09-26"), ("a-meta2", "2026-09-27"), ("a-v1", "2026-10-11"),
             ("a-v2", "2026-10-19"), ("a-v3", "2026-10-24")]
    train, held = fl.split_lists(cands, 3)
    assert held == "a-v3"
    # the 3 newest below the held-out, oldest-first
    assert train == ["a-meta2", "a-v1", "a-v2"]


def test_split_lists_tiebreak_by_stem():
    cands = [("a-v2", "2026-10-19"), ("a-v1", "2026-10-19"), ("a-meta1", "2026-09-26"),
             ("a-meta2", "2026-09-27")]
    train, held = fl.split_lists(cands, 3)
    assert held == "a-v2"  # same date: larger stem is newer
    assert train == ["a-meta1", "a-meta2", "a-v1"]


def test_split_lists_short_raises():
    with pytest.raises(ValueError):
        fl.split_lists([("a-meta1", "2026-09-26"), ("a-meta2", "2026-09-27")], 3)
    with pytest.raises(ValueError):
        fl.split_lists([], 3)
    # exactly n_train + 1 is fine
    train, held = fl.split_lists([(f"s{i}", f"2026-09-2{i}") for i in range(4)], 3)
    assert held == "s3" and len(train) == 3


# --- opponents_for: drop stems containing the archetype string ---


OPPS = ["angelechy-elfnote-meta1", "elfnote-mitsurugi-synchron-meta1", "kewl-tune-meta1",
        "onomat-constellar-tellarknights-meta1", "power-patron-meta1", "sky-striker-meta2",
        "synchron-elfnote-meta2", "trains-meta2"]


def test_opponents_for_sky_striker():
    assert "sky-striker-meta2" not in fl.opponents_for("sky-striker", OPPS)
    assert len(fl.opponents_for("sky-striker", OPPS)) == len(OPPS) - 1


def test_opponents_for_elfnote_and_mitsurugi():
    for arch in ("elfnote", "mitsurugi"):
        assert "elfnote-mitsurugi-synchron-meta1" not in fl.opponents_for(arch, OPPS)
    assert "elfnote-mitsurugi-synchron-meta1" in fl.opponents_for("sky-striker", OPPS)
    # substring rule: angelechy-elfnote-meta1 and synchron-elfnote-meta2 contain "elfnote"
    assert "angelechy-elfnote-meta1" not in fl.opponents_for("elfnote", OPPS)
    assert "synchron-elfnote-meta2" not in fl.opponents_for("elfnote", OPPS)


# --- is_excluded: URL deck id or card multiset ---


def test_is_excluded_by_id_or_multiset():
    ids = {"735631"}
    keys = {fl.multiset_key(Counter([1, 1, 2]))}
    assert fl.is_excluded("735631", Counter([9, 9]), ids, keys)  # same URL deck id
    assert fl.is_excluded("999999", Counter([1, 2, 1]), ids, keys)  # same multiset
    assert not fl.is_excluded("999999", Counter([7, 8]), ids, keys)
    assert not fl.is_excluded("7356310", Counter([7, 8]), ids, keys)  # id prefix is not a match
