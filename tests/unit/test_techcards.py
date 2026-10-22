"""Tech-card harness: slot swaps, exposure, question file (engine plan N4)."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks import load_ydk
from decklight.engine.duel import shuffle_main
from decklight.eval.techcards import (
    apply_swaps,
    card_name,
    find_code,
    opening_exposure,
    resolve_code,
    swap,
)

FIX = ROOT / "tests" / "fixtures" / "decks"
QFILE = ROOT / "docs" / "techcards" / "questions-2026-10-16.json"


def _tech_ab():
    spec = importlib.util.spec_from_file_location("tech_ab_mod", ROOT / "scripts" / "tech_ab.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tech_ab_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_swap_keeps_length_and_positions():
    d = load_ydk(FIX / "BlueEyes.ydk")
    counts = Counter(d.main)
    out_code = next(c for c, n in counts.items() if n >= 2)
    in_code = next(c for c, n in counts.items() if n == 1 and c != out_code)
    b = swap(d, out_code, in_code, 1)
    assert len(b.main) == len(d.main) and len(b.extra) == len(d.extra)
    assert Counter(b.main)[out_code] == counts[out_code] - 1
    assert Counter(b.main)[in_code] == counts[in_code] + 1
    # only the first occurrence slot changed
    diffs = [i for i, (x, y) in enumerate(zip(d.main, b.main)) if x != y]
    assert len(diffs) == 1 and d.main[diffs[0]] == out_code and b.main[diffs[0]] == in_code
    # default swaps all copies
    c = swap(d, out_code, in_code)
    assert out_code not in c.main and Counter(c.main)[in_code] == counts[in_code] + counts[out_code]


def test_swap_errors():
    d = load_ydk(FIX / "BlueEyes.ydk")
    with pytest.raises(ValueError):
        swap(d, 99999999, 10000000)  # absent
    counts = Counter(d.main)
    out_code = next(c for c, n in counts.items() if n >= 1)
    with pytest.raises(ValueError):
        swap(d, out_code, 10000000, counts[out_code] + 1)  # too many copies


def test_swap_extra_section():
    d = load_ydk(FIX / "BlueEyes.ydk")
    assert len(d.extra) > 0
    out_code = d.extra[0]
    b = swap(d, out_code, 15025844, 1)
    assert len(b.main) == len(d.main) and tuple(b.main) == tuple(d.main)
    assert b.extra.count(out_code) == list(d.extra).count(out_code) - 1


def test_shuffle_differs_only_at_swapped_positions():
    d = load_ydk(FIX / "BlueEyes.ydk")
    counts = Counter(d.main)
    out_code = next(c for c, n in counts.items() if n >= 2)
    in_code = next(c for c, n in counts.items() if n == 1 and c != out_code)
    b = swap(d, out_code, in_code, 1)
    for seed, team in ((0, 0), (0, 1), (7, 0), (123, 1)):
        sa, sb = list(d.main), list(b.main)
        shuffle_main(sa, seed, team)
        shuffle_main(sb, seed, team)
        # same permutation of lists differing at one slot: exactly one index differs
        diffs = [i for i, (x, y) in enumerate(zip(sa, sb)) if x != y]
        assert len(diffs) == 1
        i = diffs[0]
        assert (sa[i], sb[i]) == (out_code, in_code)


def test_exposure_matches_manual_shuffle():
    d = load_ydk(FIX / "BlueEyes.ydk")
    main = list(d.main)
    for seed, team in ((0, 0), (3, 1)):
        order = list(main)
        shuffle_main(order, seed, team)
        exp = opening_exposure(main, seed, team, 5)
        assert [c for _, c in exp] == order[-5:]
        assert [i for i, _ in exp] == list(range(len(order) - 5, len(order)))
        seen = opening_exposure(main, seed, team, 8)
        assert [c for _, c in seen] == order[-8:]


def test_names_resolve():
    assert find_code("Ash Blossom & Joyous Spring") == 14558127
    assert find_code("Mystical Elf") == 15025844
    assert card_name(14558127) == "Ash Blossom & Joyous Spring"
    assert resolve_code("14558127") == 14558127
    assert resolve_code(10045474) == 10045474
    with pytest.raises(ValueError):
        find_code("Not A Real Card Name xyz")


def test_question_file_parses_and_resolves():
    mod = _tech_ab()
    qs = mod.load_questions(QFILE)
    assert len(qs) == 25
    kinds = Counter(q["kind"] for q in qs)
    assert kinds["pos_control"] == 5 and kinds["neg_control"] == 5 and kinds["real"] == 15
    for q in qs:
        deck = load_ydk(ROOT / q["deck"])
        assert 40 <= len(deck.main) <= 60
        codes, _ = mod.resolve_swaps(ROOT / q["deck"], q["swaps"])
        v = apply_swaps(deck, codes)
        assert len(v.main) == len(deck.main)  # same length: identical deals except swapped cards
        for out_c, in_c, _ in codes:
            card_name(out_c)
            card_name(in_c)


def test_missing_card_fails_fast():
    """A swap naming a card absent from the deck exits non-zero, never silently."""
    mod = _tech_ab()
    with pytest.raises(SystemExit):
        mod.resolve_swaps(FIX / "BlueEyes.ydk", [["Mystical Elf", "Ash Blossom & Joyous Spring", 1]])


def test_write_ydk_roundtrip(tmp_path):
    from decklight.eval.techcards import write_ydk

    d = load_ydk(FIX / "BlueEyes.ydk")
    b = swap(d, d.main[0], 15025844, 1)
    p = write_ydk(b, tmp_path / "v.ydk")
    assert load_ydk(p) == b


def test_tiny_question_end_to_end(tmp_path):
    """A synthetic one-question file over fixture decks runs, with A/A exactly 0."""
    mod = _tech_ab()
    d = load_ydk(FIX / "BlueEyes.ydk")
    out_code = d.main[0]
    q = [{"id": "tiny", "deck": str(FIX / "BlueEyes.ydk"),
          "swaps": [[out_code, 15025844, 1]], "kind": "real"}]
    qf = tmp_path / "q.json"
    qf.write_text(json.dumps(q))
    out = tmp_path / "tech"
    rc = mod.main(["--questions", str(qf), "--pilot", "greedy", "--opp", "greedy",
                   "--opp-decks", str(FIX), "--n-opp-decks", "1",
                   "--seeds", "1", "--threads", "2", "--num-envs", "4",
                   "--out", str(out)])
    assert rc == 0
    rep = json.loads((out / "tiny" / "report.json").read_text())
    assert rep["n_pairs"] == 1 * 1 * 2 == len(rep["rows"])
    assert rep["stats"]["noise_floor"]["n"] + rep["stats"]["exposure"]["opening"]["n"] \
        + rep["stats"]["exposure"]["drawn_by_t3"]["n"] == rep["n_pairs"]
    aa = json.loads((out / "aa" / "report.json").read_text())
    assert aa["stats"]["overall"]["diff"] == 0.0
    assert (out / "tiny" / "summary.md").read_text().strip()
