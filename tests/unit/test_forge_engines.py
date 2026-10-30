"""Forge engine-novelty helpers: pure functions on Counters (no network)."""

from collections import Counter

from decklight.forge.engines import greedy_select, novelty, overlap, signature, staples


def C(*codes: int) -> Counter:
    return Counter(codes)


# --- staples ---


def test_staples_frac_threshold():
    arch_lists = {
        "a": [C(1, 10), C(1, 11)],
        "b": [C(1, 20)],
        "c": [C(1, 30)],
        "d": [C(2, 40)],
    }
    assert 1 in staples(arch_lists, frac=0.5)  # 3 of 4 archetypes
    assert 1 not in staples(arch_lists, frac=0.8)  # 0.75 < 0.8
    assert 2 not in staples(arch_lists, frac=0.5)


def test_staples_empty():
    assert staples({}) == set()


# --- signature / novelty / overlap ---


def test_signature_min_share_and_staples():
    lists = [C(1, 2), C(1, 2), C(1, 3)]
    assert signature(lists, {1}, min_share=0.5) == {2}
    assert signature(lists, {1}, min_share=0.75) == set()
    assert signature([], {1}) == set()


def test_novelty_and_overlap_edges():
    assert novelty(set(), {1, 2}) == 0.0
    assert novelty({1, 2}, set()) == 1.0
    assert novelty({1, 2}, {2, 3}) == 0.5
    assert overlap(set(), {1}) == 0.0
    assert overlap({1}, set()) == 0.0
    assert overlap({1, 2}, {2, 3, 4}) == 0.5  # |&| = 1, min size 2
    assert overlap({1}, {1, 2, 3}) == 1.0  # contained engine


# --- greedy_select ---


def test_greedy_mix_never_picked_popular_new_wins():
    covered = {1, 2, 3, 4}
    cands = {
        "mix": (100, {1, 2, 3, 4}),  # (a) union of covered engines: novelty 0
        "new-popular": (20, {10, 11, 12, 13}),  # (b) beats less popular
        "new-quiet": (5, {20, 21, 22, 23}),
    }
    picks = greedy_select(cands, covered, 2)
    assert [p["arch"] for p in picks] == ["new-popular", "new-quiet"]
    assert all(p["novelty"] == 1.0 for p in picks)


def test_greedy_near_duplicate_drops_out_and_early_stop():
    cands = {
        "a": (10, {1, 2, 3, 4}),
        "a-prime": (9, {1, 2, 3, 5}),  # (c) 3/4 shared with A -> novelty 0.25 once A is picked
        "other": (1, {1, 2}),  # novelty 0 vs covered+A
    }
    picks = greedy_select(cands, set(), 5)
    assert [p["arch"] for p in picks] == ["a"]  # (d) early stop: nothing else qualifies


def test_staples_never_named_engine_cards():
    from collections import Counter

    from decklight.forge.engines import staples

    lists = {f"a{i}": [Counter({1: 1, 2: 1})] for i in range(4)}
    assert staples(lists, frac=0.5) == {1, 2}
    assert staples(lists, frac=0.5, never={2}) == {1}
