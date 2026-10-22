"""Honest-override rule (src/decklight/search/honest.py) and its use in search_game_gen."""

import math

import numpy as np

from decklight.search.honest import HonestRule, decide, decide_pair, decisive_label, paired_gap, semantic_key

R = HonestRule(z=2.0, min_gap=0.05, s_max=8, step=4)


def test_paired_gap_uses_common_samples():
    m, se, n = paired_gap([1.0, 0.5, 0.0], [0.5, 0.0])
    assert n == 2 and math.isclose(m, 0.5) and se == 0.0
    assert paired_gap([1.0], [0.0])[1] == math.inf
    assert paired_gap([], [0.0])[2] == 0


def test_decide_keeps_pick_unless_significant():
    pick = [0.0, 0.1, -0.1, 0.0]
    noisy = [0.6, -0.4, 0.5, -0.3]  # mean +0.1 over pick, but noisy
    clear = [0.3, 0.4, 0.2, 0.3]  # +0.3, tight
    worse = [-0.5, -0.5, -0.5, -0.5]
    d = decide([pick, noisy, worse], 0, R)
    assert d.status == "undecided" and d.chosen_j == 0 and d.challenger_j == 1
    d = decide([pick, clear, worse], 0, R)
    assert d.status == "override" and d.chosen_j == 1
    d = decide([pick, worse], 0, R)
    assert d.status == "keep" and d.chosen_j == 0
    # deterministic tiny gap: SE 0 but below min_gap
    d = decide([[0.0] * 4, [0.01] * 4], 0, R)
    assert d.status == "undecided" and d.chosen_j == 0
    d = decide_pair([pick, clear, noisy], 0, 2, R)
    assert d.challenger_j == 2 and d.status == "undecided"


def test_decisive_label_needs_runner_up_gap():
    a, b, c = [0.5, 0.6, 0.55, 0.5], [0.0, 0.1, 0.05, 0.0], [-1.0] * 4
    d = decisive_label([b, a, c], R)
    assert d.status == "label" and d.chosen_j == 1 and d.challenger_j == 0
    d = decisive_label([a, [x + 0.01 for x in a]], R)
    assert d.status == "undecided"


def _obs(rows, link_on_field=False):
    acts = np.zeros((8, 10), np.int32)
    for i, r in enumerate(rows):
        acts[i, : len(r)] = r
    cards = np.zeros((4, 21), np.int32)
    mask = np.zeros(4, bool)
    if link_on_field:
        cards[0, [0, 1, 3, 7]] = [1234, 3, 0, 0x4000000]  # TYPE_LINK
        mask[0] = True
    return {"actions": acts, "action_mask": np.arange(8) < len(rows), "cards": cards, "card_mask": mask}


def test_semantic_key_merges_copies_and_zones():
    from decklight.engine.actions import Kind
    from decklight.search.honest import merge_candidates

    act = [Kind.ACTIVATE, 5, 111, 0, 1, 0, 16, 2, 0]
    act_copy = [Kind.ACTIVATE, 9, 111, 0, 1, 0, 16, 2, 0]  # another copy: differs only in row_of
    act_other = [Kind.ACTIVATE, 6, 111, 0, 2, 0, 16, 2, 0]  # another effect (desc) of the same card
    z1, z3, zc = ([Kind.PLACE, 0, 0, 0, 0, z, 0, 0, 0] for z in (1, 3, 2))
    o = _obs([act, act_copy, act_other, z1, z3, zc])
    assert semantic_key(o, 0) == semantic_key(o, 1) != semantic_key(o, 2)
    assert merge_candidates([1, 0, 2, 3, 4, 5], o) == [1, 2, 3, 5]
    ol = _obs([act, act_copy, act_other, z1, z3, zc], link_on_field=True)
    assert merge_candidates([0, 1, 2, 3, 4, 5], ol) == [0, 2, 3, 4, 5]


def test_ho_game_logs_decisions_and_overrides_only_significant(engine):
    from test_search_play import _cfg, _tiny_forward

    from decklight.engine.cpp_env import CppEnv
    from decklight.search.play import argmax_select, search_game_gen
    from decklight.search.turn1 import drive_many
    from decklight.search.window import WindowSearch

    def never(buf):
        raise AssertionError("unused")

    fw = _tiny_forward()
    cfg = _cfg()
    rule = HonestRule(z=2.0, min_gap=0.05, s_max=4, step=2)
    ws = WindowSearch(engine, never, k=3, horizon=1, samples=2, seed=0)
    gen = search_game_gen(engine, cfg, ws, 0, argmax_select, 0, own_turn1=True, conf_skip=1.01, own_frac=1.0,
                          honest=True, det_tries=4, ho=rule)
    res = dict(drive_many([(0, gen)], fw, 2))[0]
    assert not res.truncated and res.winner in (0, 1)
    assert res.log and all("ho" in e for e in res.log)
    st = [e["ho"]["status"] for e in res.log]
    assert res.n_override == st.count("override")
    for e in res.log:
        h = e["ho"]
        if h["status"] != "override":
            assert e["chosen"] == e["played"]
        if h["rounds"]:
            assert h["n"] == 4  # escalated pairs reach s_max
            assert max(len(s) for s in e["samples"]) == 4
    env = CppEnv(engine, cfg)
    obs = env.reset()
    for c in res.choices:
        obs = env.step(int(c))
    assert obs is None and int(env.winner) == res.winner
    # an unreachable floor never overrides
    ws2 = WindowSearch(engine, never, k=3, horizon=1, samples=2, seed=0)
    gen2 = search_game_gen(engine, cfg, ws2, 0, argmax_select, 0, own_turn1=True, conf_skip=1.01, own_frac=1.0,
                           ho=HonestRule(min_gap=1e9, s_max=2))
    res2 = dict(drive_many([(0, gen2)], fw, 2))[0]
    assert res2.log and res2.n_override == 0


def test_replicate_overrides_item(engine):
    """scripts/audit/replicate_overrides.py item_gen re-searches an overridden decision of a recorded game."""
    import importlib.util
    import sys
    from pathlib import Path

    from test_search_play import FIXTURES, _tiny_forward

    from decklight.decks import load_ydk
    from decklight.engine.duel import DuelConfig
    from decklight.search.play import argmax_select, search_game_gen
    from decklight.search.turn1 import drive_many
    from decklight.search.window import WindowSearch

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("replicate_overrides", root / "scripts/audit/replicate_overrides.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["replicate_overrides"] = mod
    spec.loader.exec_module(mod)

    def never(buf):
        raise AssertionError("unused")

    fw = _tiny_forward()
    paths = sorted((FIXTURES / "decks").glob("*.ydk"))
    cfg = DuelConfig(load_ydk(paths[2]), load_ydk(paths[3]), seed=8)
    ws = WindowSearch(engine, never, k=3, horizon=1, samples=2, seed=0)
    res = dict(drive_many([(0, search_game_gen(engine, cfg, ws, 0, argmax_select, 0, own_turn1=True, conf_skip=1.01,
                                               own_frac=1.0))], fw, 2))[0]
    rec = {"decks": [paths[2].stem, paths[3].stem], "seed": 8, "s_seat": 0, "game_idx": 0, "choices": res.choices}
    ov = [e for e in res.log if e["chosen"] != e["played"]]
    assert ov, "no override in the fixture game"
    ws2 = WindowSearch(engine, never, k=2, horizon=1, samples=4, seed=1)
    out = dict(drive_many([(0, mod.item_gen(engine, ws2, rec, ov[0], FIXTURES / "decks", 4, 8, 0))], fw, 2))[0]
    assert "error" not in out, out
    assert out["n"] >= 2 and out["holds"] == (out["gap"] > 0)


def test_replicate_hw_label_item(engine, tmp_path):
    """replicate_overrides --hw: a stage L record (stored config + actors, no s_seat) replays and re-searches a
    decisive label (best vs runner-up) at the acting player's seat; hw_labels reads gen_window --ho records."""
    import importlib.util
    import json
    import sys
    from pathlib import Path

    from test_search_play import FIXTURES, _tiny_forward

    from decklight.decks import load_ydk
    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import DuelConfig
    from decklight.search.turn1 import drive_many
    from decklight.search.window import WindowSearch

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("replicate_overrides", root / "scripts/audit/replicate_overrides.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["replicate_overrides"] = mod
    spec.loader.exec_module(mod)

    paths = sorted((FIXTURES / "decks").glob("*.ydk"))
    d0, d1 = load_ydk(paths[2]), load_ydk(paths[3])
    cfg = DuelConfig(d0, d1, seed=8)
    env = CppEnv(engine, cfg, True)
    obs, choices, actors, k_at = env.reset(), [], [], None
    while obs is not None and len(choices) < 200:
        legal = [int(i) for i in obs["action_mask"].nonzero()[0]]
        if k_at is None and len(legal) >= 2 and int(obs["global"][2]) >= 2:
            k_at, pair, turn = len(choices), legal[:2], int(obs["global"][2])
        actors.append(int(env.player))
        choices.append(legal[0])
        obs = env.step(legal[0])
    assert k_at is not None
    conf = {"deck0_main": list(d0.main), "deck0_extra": list(d0.extra), "deck1_main": list(d1.main),
            "deck1_extra": list(d1.extra), "seed": 8, "hand": cfg.hand, "draw": cfg.draw, "lp": cfg.lp, "setup": [],
            "shuffle": cfg.shuffle}  # fmt: skip
    entry = {"k": k_at, "kind": "own", "turn": turn, "cands": pair, "chosen": pair[1], "played": pair[0],
             "fired": False, "ho": {"status": "label", "best": pair[1], "runner_up": pair[0]}}  # fmt: skip
    rec = {"config": conf, "auto_single": True, "choices": choices, "actors": actors, "search": [entry]}
    (tmp_path / "p0").mkdir()
    (tmp_path / "p0" / "x.jsonl").write_text(json.dumps(rec) + "\n")
    items = mod.hw_labels(tmp_path)
    assert len(items) == 1 and items[0][1]["played"] == pair[0] and items[0][1]["chosen"] == pair[1]

    def never(buf):
        raise AssertionError("unused")

    ws = WindowSearch(engine, never, k=2, horizon=1, samples=4, seed=1)
    r, e = items[0]
    out = dict(drive_many([(0, mod.item_gen(engine, ws, r, e, FIXTURES / "decks", 4, 8, 0))], _tiny_forward(), 2))[0]
    assert "error" not in out, out
    assert out["n"] >= 2 and out["pick"] == pair[0] and out["chosen"] == pair[1]
