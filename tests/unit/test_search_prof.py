"""Profiling instrumentation (speed plan M0): --prof is bit-identical, latency mode records walls."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"

PROF_KEYS = ("t_det_init", "t_det", "n_prop", "n_acc", "t_setup", "t_env", "t_py", "t_wait",
             "rounds", "obs_fwd", "rollout_len", "t_value", "t_wall")


def _tiny_forward():
    """A tiny d=16 PolicyNet on synthetic embeddings (as in test_search_play.py), as a batched logits fn."""
    from ygo_zero.decks import load_ydk
    from ygo_zero.model.bc_data import OBS_KEYS
    from ygo_zero.model.policy import PolicyNet

    decks = [load_ydk(p) for p in sorted((FIXTURES / "decks").glob("*.ydk"))]
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    net = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).eval()

    @torch.no_grad()
    def forward(obs_list):
        t = {k: torch.from_numpy(np.ascontiguousarray(np.stack([o[k] for o in obs_list])))
             for k in OBS_KEYS}
        return net(t).numpy()

    return forward


def _cfg():
    from ygo_zero.decks import load_ydk
    from ygo_zero.engine.duel import DuelConfig

    decks = [load_ydk(p) for p in sorted((FIXTURES / "decks").glob("*.ydk"))]
    return DuelConfig(decks[2], decks[3], seed=8)


def _play(engine, forward, prof=False, latency=False, **kw):
    from ygo_zero.search.play import argmax_select, search_game_gen
    from ygo_zero.search.turn1 import drive_many
    from ygo_zero.search.window import WindowSearch

    def never(buf):
        raise AssertionError("unused")

    cfg = _cfg()
    ws = WindowSearch(engine, never, k=kw.pop("k", 2), horizon=1, samples=1, seed=0, prof=prof)
    gen = search_game_gen(engine, cfg, ws, kw.pop("s_seat", 0), argmax_select, kw.pop("game_seed", 0),
                          own_turn1=True, prof=prof, latency=latency, **kw)
    return dict(drive_many([(0, gen)], forward, 2))[0]


def _honest_kw():
    return {"conf_skip": 1.01, "own_frac": 1.0, "honest": True, "det_tries": 2, "det_fallback": "true"}


def test_prof_is_bit_identical(engine):
    """prof off vs on over a few honest deals: choices, samples, scores and det counters identical; the prof dict
    (with all timing keys) and full scores_all present only when on. max_searches=4 keeps it fast: the first 4
    searches are fully timed, the rest plays raw."""
    fw = _tiny_forward()
    for seed in (0, 1):
        off = _play(engine, fw, game_seed=seed, max_searches=4, **_honest_kw())
        on = _play(engine, fw, prof=True, game_seed=seed, max_searches=4, **_honest_kw())
        assert off.choices == on.choices
        assert off.actors == on.actors
        assert (off.n_searched, off.n_override, off.det_proposals, off.det_accepted,
                off.det_fallbacks, off.det_skipped) == \
               (on.n_searched, on.n_override, on.det_proposals, on.det_accepted,
                on.det_fallbacks, on.det_skipped)
        assert on.n_searched >= 1 and on.det_proposals > 0
        assert on.search_cut  # the cap engaged: the timed prefix is what we compare
        assert len(off.log) == len(on.log) == on.n_searched
        for e_off, e_on in zip(off.log, on.log, strict=True):
            assert [e_off["samples"], e_off["scores"]] == [e_on["samples"], e_on["scores"]]
            assert "prof" not in e_off
            assert set(e_on["scores_all"]) == {"board@h1", "done@h1", "cards@h1"}
            assert set(e_off["scores_all"]) == {"board@h1", "done@h1", "cards@h1"}
            p = e_on["prof"]
            assert tuple(p) == PROF_KEYS
            assert p["t_wall"] > 0 and p["rounds"] >= 1 and p["obs_fwd"] >= 1
            assert isinstance(p["rollout_len"], list) and p["rollout_len"]
            assert {k: v for k, v in e_on.items() if k != "prof"} == e_off


def test_latency_records_walls(engine):
    """latency mode on a full non-honest game: every decision lands in lat_raw or lat_search, searched moves match
    n_searched, own turns span."""
    res = _play(engine, _tiny_forward(), latency=True, game_seed=0, conf_skip=1.01, own_frac=1.0)
    assert len(res.lat_raw) + len(res.lat_search) == len(res.choices)
    assert len(res.lat_search) == res.n_searched >= 1
    assert all(w >= 0 for w in res.lat_raw + res.lat_search + res.lat_turn)
    assert res.lat_turn  # own_turn1=True: S has timed own turns


def _synthetic_dir(tmp_path: Path) -> Path:
    d = tmp_path / "arm"
    d.mkdir()
    p = {"t_det_init": 0.05, "t_det": 0.6, "n_prop": 57, "n_acc": 4, "t_setup": 0.4, "t_env": 0.1,
         "t_py": 0.05, "t_wait": 0.2, "rounds": 3, "obs_fwd": 36, "rollout_len": [4, 4, 4, 4],
         "t_value": 0.01, "t_wall": 1.5}
    games = [
        {"decks": ["a", "b"], "seed": 9000, "s_seat": 0, "game_idx": 0, "winner": 0, "truncated": False,
         "turns": 6, "choices": [0, 1], "actors": [0, 1], "n_searched": 2, "n_override": 1,
         "lat_raw": [0.01, 0.02], "lat_search": [1.5, 1.6], "lat_turn": [3.1],
         "search": [{"k": 3, "seat": 0, "kind": "own", "turn": 2, "cands": [0, 1], "samples": [[0.5], [0.6]],
                     "chosen": 1, "played": 0, "prof": p},
                    {"k": 9, "seat": 0, "kind": "window", "turn": 3, "cands": [1, 0], "samples": [[0.7], [0.4]],
                     "chosen": 1, "played": 1, "prof": p}]},
        {"decks": ["b", "a"], "seed": 9001, "s_seat": 1, "game_idx": 1, "winner": 1, "truncated": False,
         "turns": 5, "choices": [1, 0], "actors": [0, 1], "n_searched": 1, "n_override": 0,
         "lat_raw": [0.015], "lat_search": [1.4], "lat_turn": [1.4],
         "search": [{"k": 5, "seat": 1, "kind": "own", "turn": 2, "cands": [0, 1], "samples": [[0.2], [0.1]],
                     "chosen": 0, "played": 0, "prof": p}]},
    ]
    with (d / "games-0.jsonl").open("w") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")
    (d / "prof-0.json").write_text(json.dumps({
        "calls": 100, "hist": {"12": 60, "36": 40}, "t_batch": 5.0, "t_fwd": 50.0, "t_gpu": 30.0,
        "t_sync": 20.0, "t_h2d": 0.0, "t_d2h": 0.0, "t_ygo": 1.0, "proc": 0, "process_time": 500.0,
        "wall": 600.0, "threads": 3, "torch_threads": 1, "thread_cpu": {"1": 400.0},
        "cuda_reserved": None, "games": 2, "searched": 3,
    }))
    return d


def test_reports_on_synthetic(tmp_path):
    """prof_report.py and latency_report.py print their tables on a synthetic arm."""
    d = _synthetic_dir(tmp_path)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/audit/prof_report.py"), str(d)],
                       capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0, r.stderr
    for token in ("t_det", "t_setup", "t_env", "t_py", "t_wait", "t_value", "t_wall",
                  "unattributed", "box-minute", "forwards"):
        assert token in r.stdout, token
    r = subprocess.run([sys.executable, str(ROOT / "scripts/audit/latency_report.py"), str(d)],
                       capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0, r.stderr
    for token in ("lat_raw", "lat_search", "lat_turn"):
        assert token in r.stdout, token
