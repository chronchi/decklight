"""R1: search on clones (Rust core) = search on prefix replays, A/A.

`cpp_env.SEARCH_CLONE` (YGO_CORE=rust) makes WindowSearch build each branch as a clone of one root env
(per world, kept by the Determinizer) stepped by its candidate, instead of replaying the whole prefix per branch.
Both paths must give the same game: same choices, same per-decision scores and samples. Rust-only, like
test_rust_core.py (the two extensions cannot share a process): run this file on its own.
"""

import os
import sys
import time

import pytest

if "decklight._ygoenv" in sys.modules:
    pytest.skip("default decklight._ygoenv already loaded; run this file standalone", allow_module_level=True)

os.environ["YGO_CORE"] = "rust"
pytest.importorskip("decklight._ygoenv_rs", reason="Rust extension not built (scripts/build_rust_core.sh)")

from test_search_play import _cfg, _tiny_forward

from decklight.engine import cpp_env
from decklight.engine.duel import Engine
from decklight.search import play as play_mod
from decklight.search.honest import HonestRule
from decklight.search.play import argmax_select, search_game_gen
from decklight.search.turn1 import drive_many
from decklight.search.window import WindowSearch


@pytest.fixture(scope="module")
def engine():
    return Engine()


def _game(engine, fw, clone: bool, **kw):
    cpp_env.SEARCH_CLONE = clone
    play_mod.SEARCH_CLONE = clone

    def never(buf):
        raise AssertionError("unused")

    ws = WindowSearch(engine, never, k=3, horizon=1, samples=kw.pop("samples", 2), seed=0)
    gen = search_game_gen(engine, _cfg(), ws, 0, argmax_select, 0, own_turn1=True, conf_skip=1.01,
                          own_frac=1.0, max_searches=15, **kw)
    t0 = time.perf_counter()
    res = dict(drive_many([(0, gen)], fw, 2))[0]
    return res, time.perf_counter() - t0


@pytest.mark.parametrize("mode", ["perfect", "honest", "honest_ho"])
def test_clone_search_matches_replay_search(engine, mode):
    fw = _tiny_forward()  # one net for both arms (its init is not seeded: compare within a process only)
    kw = {"perfect": {}, "honest": {"honest": True, "det_tries": 4},
          "honest_ho": {"honest": True, "det_tries": 4, "ho": HonestRule(z=2.0, min_gap=0.05, s_max=4, step=2)}}[mode]
    try:
        a, ta = _game(engine, fw, False, **dict(kw))
        b, tb = _game(engine, fw, True, **dict(kw))
    finally:
        cpp_env.SEARCH_CLONE = play_mod.SEARCH_CLONE = True
    assert a.n_searched >= 1
    assert a.choices == b.choices and a.winner == b.winner
    assert a.n_searched == b.n_searched and a.n_override == b.n_override
    assert len(a.log) == len(b.log)
    for x, y in zip(a.log, b.log, strict=True):
        assert {k: v for k, v in x.items() if "_s" not in k} == {k: v for k, v in y.items() if "_s" not in k}
    print(f"\n{mode}: searched={a.n_searched} replay {ta:.2f}s search {a.search_s:.2f}s | "
          f"clone {tb:.2f}s search {b.search_s:.2f}s")
