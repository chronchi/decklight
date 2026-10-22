"""Turn-1 search teacher (src/decklight/search/turn1.py, scripts/gen_search.py) with the deterministic stand-in policy."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen_search = _load("gen_search")


def _configs(n: int):
    from decklight.decks import load_ydk
    from decklight.engine.duel import DuelConfig

    decks = [load_ydk(f) for f in sorted((FIXTURES / "decks").glob("*.ydk"))]
    return [DuelConfig(decks[i % len(decks)], decks[(i + 1) % len(decks)], seed=100 + i) for i in range(n)]


def test_search_never_worse_than_its_policy(engine):
    """Rollout improvement: with a deterministic policy, the searched turn ends with a board >= the policy's own."""
    from decklight.search.turn1 import Turn1Search

    s = Turn1Search(engine, gen_search.greedy_logits, k=3)
    for cfg in _configs(4):
        g, r = s.greedy(cfg), s.run(cfg)
        assert r.end_score >= g - 1e-9, (cfg.seed, g, r.end_score)
        assert len(r.choices) == len(r.actors) == len(r.labels)
        assert {lab for lab, a in zip(r.labels, r.actors, strict=True) if a == 0} <= {"search"}
        assert r.overrides <= r.searched <= r.decisions


def test_k1_is_the_policy(engine):
    from decklight.search.turn1 import Turn1Search

    s = Turn1Search(engine, gen_search.greedy_logits, k=1)
    for cfg in _configs(2):
        r = s.run(cfg)
        assert r.searched == 0 and r.overrides == 0
        assert abs(r.end_score - s.greedy(cfg)) < 1e-9


def test_records_replay_and_map_to_search_kind(engine, tmp_path):
    from decklight.demos.dataset import iter_samples, record_config
    from decklight.engine.cpp_env import CppEnv
    from decklight.search.turn1 import board_score

    out = tmp_path / "search"
    old = sys.argv
    sys.argv = ["gen_search", "--policy", "greedy", "--hands", "3", "--k", "3", "--seed", "5", "--out", str(out)]
    try:
        gen_search.main()
    finally:
        sys.argv = old
    recs = [json.loads(x) for f in sorted(out.glob("*.jsonl")) for x in f.read_text().splitlines() if x.strip()]
    assert len(recs) == 3
    for rec in recs:
        env = CppEnv(engine, record_config(rec), rec["auto_single"])
        obs = env.reset()
        for c in rec["choices"]:
            assert obs is not None
            obs = env.step(c)
        assert obs is None or int(obs["global"][2]) > 1, "a record must cover the whole turn 1"
        if obs is not None:
            assert abs(board_score(obs, 0 if env.player == 0 else 1) - rec["end_score"]) < 1e-9
    samples = list(iter_samples(engine, out))
    n0 = sum(a == 0 for r in recs for a in r["actors"])
    assert len(samples) == n0
    assert {m["kind"] for _, _, _, m in samples} == {"search"}
    assert all(m["me"] == 0 for _, _, _, m in samples)
    assert all(int(np.asarray(o["action_mask"])[a]) for o, a, _, _ in samples)


def test_run_many_matches_one_by_one(engine):
    """The batched driver gives the same results as driving each hand alone (any completion order)."""
    from decklight.search.turn1 import Turn1Search

    s = Turn1Search(engine, gen_search.greedy_logits, k=3, batch=7)
    cfgs = _configs(5)
    many = dict(s.run_many(((i, s.both_gen(c)) for i, c in enumerate(cfgs)), width=3))
    assert sorted(many) == list(range(5))
    for i, c in enumerate(cfgs):
        g, r = many[i]
        assert g == s.greedy(c)
        one = s.run(c)
        assert (r.choices, r.end_score, r.overrides) == (one.choices, one.end_score, one.overrides)
