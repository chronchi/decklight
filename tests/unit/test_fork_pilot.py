"""Fork pilot (scripts/audit/fork_pilot.py): split-sample headroom, category tagging,
last_samples_all consistency. No GPU, no rollouts beyond tiny engine replays."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("fork_pilot", ROOT / "scripts/audit/fork_pilot.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fork_pilot"] = mod
    spec.loader.exec_module(mod)
    return mod


fp = _load()


def test_split_headroom_identical_is_zero():
    end = [[0.5] * 6, [0.5] * 6, [0.5] * 6]
    assert fp.split_headroom(end) == pytest.approx(0.0)


def test_split_headroom_clear_gain():
    end = [[0.0] * 8, [0.5] * 8]
    assert fp.split_headroom(end, 0) == pytest.approx(0.5)


def test_split_headroom_noise_is_unbiased():
    rng = np.random.default_rng(7)
    end = [list(rng.choice([-1.0, 1.0], size=400)) for _ in range(4)]
    assert abs(fp.split_headroom(end, 0)) < 0.2


def _obs(rows, msg=16, turn=3, mine_turn=True):
    acts = np.zeros((8, 10), np.int32)
    for i, r in enumerate(rows):
        acts[i, : len(r)] = r
    g = np.zeros(24, np.int32)
    g[2], g[5], g[21] = turn, int(mine_turn), msg
    return {"actions": acts, "action_mask": np.arange(8) < len(rows), "global": g}


def test_tag_ht_window():
    from ygo_zero.engine.actions import Kind

    act_hand = [Kind.ACTIVATE, 5, 111, 0, 1, 0, 16, 2, 0, 0]  # loc 2 = hand
    act_field = [Kind.ACTIVATE, 6, 222, 0, 1, 0, 16, 3, 0, 0]  # loc 3 = field
    passage = [Kind.PASS, 0, 0, 0, 0, 0, 16, 0, -1, 0]
    assert fp.tag_decision(_obs([act_hand, passage])) == {"ht", "any"}
    assert fp.tag_decision(_obs([act_field, passage])) == {"any"}


def test_tag_place_and_t1_and_single():
    from ygo_zero.engine import constants as C
    from ygo_zero.engine.actions import Kind
    from ygo_zero.engine.obs import LOC_ID

    assert LOC_ID[C.LOCATION_HAND] == 2
    z1 = [Kind.PLACE, 0, 0, 0, 0, 1, 18, 0, 0, 0]
    z2 = [Kind.PLACE, 0, 0, 0, 0, 3, 18, 0, 0, 0]
    assert fp.tag_decision(_obs([z1, z2], msg=18)) == {"place", "any"}
    s1 = [Kind.SUMMON, 4, 111, 0, 0, 0, 11, 3, 0, 0]
    s2 = [Kind.SUMMON, 5, 222, 0, 0, 0, 11, 3, 0, 0]
    assert fp.tag_decision(_obs([s1, s2], msg=11, turn=1)) == {"t1", "any"}
    assert fp.tag_decision(_obs([s1])) == set()


def test_last_samples_all_matches_last_samples(engine, decks):
    """Tiny engine rollouts_gen (no value net): last_samples == last_samples_all[select key],
    and the per-key means match the returned averages."""
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.engine.duel import DuelConfig
    from ygo_zero.search.turn1 import drive_many
    from ygo_zero.search.window import WindowSearch

    names = sorted(decks)
    cfg = DuelConfig(decks[names[0]], decks[names[1]], seed=1)
    env = CppEnv(engine, cfg, True)
    obs = env.reset()
    rng = np.random.default_rng(0)
    prefix = []
    for _ in range(6):
        if obs is None:
            break
        legal = np.flatnonzero(obs["action_mask"])
        a = int(rng.choice(legal))
        prefix.append(a)
        obs = env.step(a)
    assert obs is not None
    legal = np.flatnonzero(obs["action_mask"])
    assert len(legal) >= 2
    cands = [int(legal[0]), int(legal[1])]
    n_s = 2
    branches = [(prefix + [x], j, x, s) for j, x in enumerate(cands) for s in range(n_s)]

    def never(buf):
        raise AssertionError("unused")

    ws = WindowSearch(engine, never, k=2, horizons=(1, 99), samples=n_s, seed=0)
    gen = ws.rollouts_gen(
        cfg,
        True,
        branches,
        int(env.player),
        turn_start=int(obs["global"][2]),
        game_seed=5,
        decision_k=len(prefix),
    )
    out = dict(drive_many([(0, gen)], lambda rows: np.zeros((len(rows), 64)), 2))[0]
    assert isinstance(out, dict)
    assert ws.last_samples == ws.last_samples_all["board@h1"]
    for kk, col in out.items():
        for j in range(2):
            assert col[j] == pytest.approx(float(np.mean(ws.last_samples_all[kk][j])))
            assert len(ws.last_samples_all[kk][j]) == n_s


def test_summarize_rejects_all_error_rows():
    stats = {"cats": {c: {"dpg": 1.0} for c in ("ht", "place", "t1", "any")}}
    rows = [{"game_idx": 0, "k": 1, "tags": ["any"], "error": "only 0 worlds"}]
    with pytest.raises(SystemExit):
        fp.summarize(rows, stats)
