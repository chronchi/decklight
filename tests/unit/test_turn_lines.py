"""Whole-turn lines from any start state, with an active opponent (CPU, tiny d=16 random policy)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from test_turn1_lines import _deck_names, _make_ckpts

from decklight.demos.dataset import iter_samples
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig
from decklight.eval.turn1 import (
    actor_field_of,
    config_to_dict,
    dags_config,
    replay_prefix,
    replay_turn,
    turn_player_of,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _run_turn1(mod, ck: dict, out: Path, opp: str = "policy") -> int:
    ydks = ",".join(str(FIXTURES / f"{n}.ydk") for n in _deck_names())
    return mod.main(["--ckpt", str(ck["ckpt"]), "--emb", str(ck["emb"]),
                     "--starts", "turn1", "--decks", ydks, "--n", "2", "--k", "2",
                     "--seed", "0", "--out", str(out), "--opp", opp,
                     "--procs", "1", "--threads", "1", "--device", "cpu"])


def _replay_field(engine, row: dict) -> dict:
    from decklight.demos.dataset import record_config

    env = CppEnv(engine, record_config({"config": row["config"]}), row.get("auto_single", True))
    obs = env.reset()
    for c in [int(c) for c in row["choices"]]:
        if obs is None:
            break
        if int(obs["global"][2]) > int(row["turn_start"]):
            break
        obs = env.step(c)
    assert obs is not None, "line ended the duel before its turn ended"
    return {"field": actor_field_of(obs, env.player, int(row["actor"]))}


def test_turn1_active_opponent_rows_replay(engine, tmp_path):
    mod = _load("turn_lines")
    ck = _make_ckpts(tmp_path / "ck")
    out = tmp_path / "out"
    assert _run_turn1(mod, ck, out) == 0
    rows = _rows(out / "lines.jsonl")
    assert len(rows) == 2 * 2 * (1 + 2), f"decks * hands * (greedy + K), got {len(rows)}"
    assert {r["source"] for r in rows} == {"turn1"}
    assert all(r["actor"] == 0 and r["turn_start"] == 1 and r["prefix_len"] == 0 for r in rows)
    assert all(r["train_ok"] is False for r in rows)  # custom ydk list: never training data
    assert all("v_p0" not in r and "v_actor" not in r for r in rows), "no critic -> no value fields"
    for r in rows:  # the full choices replay to the same actor field in a fresh env
        assert _replay_field(engine, r)["field"] == r["field"], (r["deck"], r["hand"], r["kind"])
        got = replay_turn(engine, __import__("decklight.demos.dataset", fromlist=["record_config"]).record_config(
            {"config": r["config"]}), [int(c) for c in r["choices"]], int(r["actor"]), int(r["turn_start"]),
            r.get("auto_single", True))
        assert got["field"] == r["field"]
        assert got["actor_decisions"] == r["actor_decisions"]
    if not any(r["opp_options"] > 0 for r in rows):
        pytest.skip("tiny random-init policy never gave the opponent a real choice here")
    assert any(r["opp_decisions"] > 0 for r in rows), "the active opponent must act in some run"


def _play_game(engine, names, seed: int):
    """A short real game with rule-based play; returns (config dict, choices, turns) or None."""
    from decklight.decks import load_ydk
    from decklight.demos.game import Responder, _opts

    d0, d1 = (load_ydk(FIXTURES / f"{n}.ydk") for n in names[:2])
    cfg = DuelConfig(d0, d1, seed=seed)
    env = CppEnv(engine, cfg)
    obs = env.reset()
    rsp = Responder(seed)
    choices, turns = [], []
    for _ in range(600):
        if obs is None or int(obs["global"][2]) > 3:
            break
        if not len(_opts(obs)):
            break
        c = int(rsp(obs))
        choices.append(c)
        turns.append(int(obs["global"][2]))
        obs = env.step(c)
    if not any(t >= 3 for t in turns):
        return None
    # mains in final (shuffled) order, like dags.start_record: the replay uses shuffle=False
    return {"deck0_main": cfg.main_of(0), "deck0_extra": list(d0.extra),
            "deck1_main": cfg.main_of(1), "deck1_extra": list(d1.extra),
            "seed": seed}, choices, turns


def _make_starts(engine, tmp_path: Path) -> Path:
    from decklight.decks import load_ydk

    names = _deck_names()
    recs = []
    for seed in range(30):
        got = _play_game(engine, names, seed)
        if got is None:
            continue
        cd, choices, turns = got
        # cut at the first decision of turn 2 and of turn 3 (prefix ends at a new turn)
        for want in (2, 3):
            k = next((i for i, t in enumerate(turns) if t == want), None)
            if k is None or k < 1:
                continue
            recs.append({"main0": cd["deck0_main"], "extra0": cd["deck0_extra"],
                         "main1": cd["deck1_main"], "extra1": cd["deck1_extra"],
                         "seed": cd["seed"], "hand": 5, "draw": 1, "lp": 8000, "setup": [],
                         "prefix": choices[:k], "turn": want, "game_id": f"g{seed}",
                         "decks": [names[0], names[1]], "src": "tiny"})
        if len(recs) >= 4:
            break
    assert len(recs) >= 2, "rule-based play must reach turn 2+ in some seed"
    assert {r["turn"] for r in recs} >= {2, 3}, f"need turn-2 and turn-3 prefixes, got {recs}"
    p = tmp_path / "starts.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def test_dags_starts_actor_is_turn_player(engine, tmp_path):
    mod = _load("turn_lines")
    ck = _make_ckpts(tmp_path / "ck")
    starts = _make_starts(engine, tmp_path)
    ydks = ",".join(str(FIXTURES / f"{n}.ydk") for n in _deck_names())
    out = tmp_path / "out"
    rc = mod.main(["--ckpt", str(ck["ckpt"]), "--emb", str(ck["emb"]),
                   "--starts", f"dags:{starts}", "--decks", ydks, "--n", "3", "--k", "1",
                   "--seed", "0", "--out", str(out),
                   "--procs", "1", "--threads", "1", "--device", "cpu"])
    assert rc == 0
    rows = _rows(out / "lines.jsonl")
    assert rows and all(r["source"] == "dags" for r in rows)
    assert {r["actor"] for r in rows} == {0, 1}, "turn-2 and turn-3 prefixes give both actors"
    for r in rows:
        env, obs = replay_prefix(engine, dags_config(
            {"main0": r["config"]["deck0_main"], "extra0": r["config"]["deck0_extra"],
             "main1": r["config"]["deck1_main"], "extra1": r["config"]["deck1_extra"],
             "seed": r["config"]["seed"], "setup": r["config"].get("setup", [])}),
            [int(c) for c in r["choices"][:r["prefix_len"]]])
        assert int(obs["global"][2]) == r["turn_start"]
        assert turn_player_of(obs, env.player) == r["actor"], "actor is the turn player"
        assert [int(c) for c in r["choices"][: r["prefix_len"]]] == \
            [int(c) for c in json.loads(starts.read_text().splitlines()[r["hand"]])["prefix"]]
        assert _replay_field(engine, r)["field"] == r["field"]


def test_dags_deck_filter_and_train_ok(tmp_path):
    mod = _load("turn_lines")
    rows = [{"decks": ["a", "b"], "turn": 2, "prefix": [0], "main0": [], "extra0": [],
             "main1": [], "extra1": [], "seed": 1},
            {"decks": ["a", "holdout-x"], "turn": 3, "prefix": [0], "main0": [], "extra0": [],
             "main1": [], "extra1": [], "seed": 2},
            {"decks": ["", ""], "turn": 2, "prefix": [0], "main0": [], "extra0": [],
             "main1": [], "extra1": [], "seed": 3},
            {"decks": ["a", "b"], "turn": 6, "prefix": [0], "main0": [], "extra0": [],
             "main1": [], "extra1": [], "seed": 4}]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    sampled, pool, skipped = mod.load_dags_starts(p, {"a", "b"}, 10, 0, None, None)
    assert skipped == 1, "empty-decks rows are skipped"
    assert len(sampled) == 2, "the holdout-deck row is dropped by the train split filter"
    assert all(set(r["decks"]) <= {"a", "b"} for r in sampled)
    # turn filters (pool after the deck filter: turns 2 and 6)
    assert len(mod.load_dags_starts(p, {"a", "b"}, 10, 0, 3, None)[0]) == 1
    assert len(mod.load_dags_starts(p, {"a", "b"}, 10, 0, None, 5)[0]) == 1
    assert len(mod.load_dags_starts(p, {"a", "b"}, 10, 0, 2, 6)[0]) == 2
    # stratification covers every bucket when asked
    many = [{"decks": ["a", "b"], "turn": t, "prefix": [0], "main0": [], "extra0": [],
             "main1": [], "extra1": [], "seed": t} for t in ([2] * 4 + [3] * 4 + [7] * 4)]
    p2 = tmp_path / "s2.jsonl"
    p2.write_text("\n".join(json.dumps(r) for r in many) + "\n")
    assert {r["turn"] for r in mod.load_dags_starts(p2, {"a", "b"}, 6, 0, None, None)[0]} >= {2, 3, 7}


def test_score_actor1_view(engine, tmp_path):
    tmod = _load("turn_lines")
    ck = _make_ckpts(tmp_path / "ck")
    starts = _make_starts(engine, tmp_path)
    ydks = ",".join(str(FIXTURES / f"{n}.ydk") for n in _deck_names())
    lines = tmp_path / "lines"
    assert tmod.main(["--ckpt", str(ck["ckpt"]), "--emb", str(ck["emb"]),
                      "--starts", f"dags:{starts}", "--decks", ydks, "--n", "2", "--k", "1",
                      "--seed", "0", "--out", str(lines),
                      "--procs", "1", "--threads", "1", "--device", "cpu"]) == 0
    rows = _rows(lines / "lines.jsonl")
    a1 = [r for r in rows if r["actor"] == 1]
    assert a1, "need actor-1 rows for the view check"
    for r in rows:
        r["train_ok"] = True
    scored_in = tmp_path / "in.jsonl"
    scored_in.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    mod = _load("score_lines")
    out = tmp_path / "scored.jsonl"
    assert mod.main(["--ckpt", str(ck["ckpt"]), "--emb", str(ck["emb"]),
                     "--lines", str(scored_in), "--out", str(out),
                     "--cands", "2", "--samples", "2", "--horizon", "2",
                     "--max-rollout", "200", "--procs", "1", "--threads", "1",
                     "--device", "cpu", "--batch", "16", "--width", "4"]) == 0
    scored = _rows(out)
    assert scored and all(np.isfinite(r["rollout"]) for r in scored)
    # sign check: the board score from player 1's view is the negation of player 0's view
    from decklight.search.window import position_score

    g = np.zeros(24, np.int32)
    g[0], g[1] = 8000, 7000
    g[6:13] = [30, 3, 0, 0, 0, 0, 10]
    g[13:20] = [28, 5, 2, 1, 1, 0, 10]
    obs = {"global": g}
    assert position_score(obs, 0, 1) == -position_score(obs, 0, 0)
    assert position_score(obs, 0, 1) > 0, "player 1 ahead on cards must score positive from its view"


def test_select_actor1_labels_self_only_post_prefix(engine, tmp_path):
    tmod = _load("turn_lines")
    ck = _make_ckpts(tmp_path / "ck")
    starts = _make_starts(engine, tmp_path)
    ydks = ",".join(str(FIXTURES / f"{n}.ydk") for n in _deck_names())
    lines = tmp_path / "lines"
    assert tmod.main(["--ckpt", str(ck["ckpt"]), "--emb", str(ck["emb"]),
                      "--starts", f"dags:{starts}", "--decks", ydks, "--n", "2", "--k", "1",
                      "--seed", "0", "--out", str(lines),
                      "--procs", "1", "--threads", "1", "--device", "cpu"]) == 0
    rows = _rows(lines / "lines.jsonl")
    a1 = [r for r in rows if r["actor"] == 1]
    assert a1, "need actor-1 rows"
    key = (a1[0]["deck"], a1[0]["source"], a1[0]["start"])
    grp = [r for r in rows if (r["deck"], r["source"], r["start"]) == key]
    gre = [r for r in grp if r["kind"] == "greedy"]
    smp = [r for r in grp if r["kind"] == "sample"]
    assert gre and smp
    gre[0]["rollout"], gre[0]["rollout_samples"] = 1.0, [1.0]
    smp[0]["rollout"], smp[0]["rollout_samples"] = 5.0, [5.0]
    smp[0]["rollout_done"], gre[0]["rollout_done"] = 0.0, 0.0
    smp[0]["rollout_capped"], gre[0]["rollout_capped"] = 0, 0
    for r in rows:
        r["train_ok"] = True
    edited = tmp_path / "ed.jsonl"
    edited.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    sel = _load("select_turn1_lines")
    keep = tmp_path / "selfimit"
    assert sel.main(["--lines", str(edited), "--out", str(keep), "--by", "rollout"]) == 0
    recs = [json.loads(line) for f in keep.glob("p1-*.jsonl")
            for line in f.read_text().splitlines() if line.strip()]
    assert recs, "the actor-1 winner must be written"
    rec = recs[0]
    pre, actor = rec["prefix_len"], rec["actor"]
    assert actor == 1 and pre > 0
    assert all(lab == "opp" for lab in rec["labels"][:pre]), "prefix decisions are context only"
    post = [(a, lab) for a, lab in zip(rec["actors"][pre:], rec["labels"][pre:])]
    assert any(a == 1 and lab == "self" for a, lab in post)
    assert all((lab == "self") == (a == 1) for a, lab in post)
    assert rec["decks"][1] == gre[0]["deck"], "decks stay in seat order (actor is seat 1)"
    samples = [(o, act, w, m) for o, act, w, m in iter_samples(engine, keep)]
    assert samples, "the selected record must replay through iter_samples"
    assert {m["kind"] for _, _, _, m in samples} == {"selfimit"}
