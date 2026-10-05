"""Mid-game window search (src/ygo_zero/search/window.py, scripts/gen_window.py) with the deterministic stand-in policy."""

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
gen_window = _load("gen_window")


def _random_games(engine, tmp_path: Path, n: int = 3) -> Path:
    """Full random-policy games in gold record format (config, choices, actors, turns)."""
    from ygo_zero.decks import load_ydk
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.engine.duel import DuelConfig

    decks = sorted((FIXTURES / "decks").glob("*.ydk"))
    out = tmp_path / "src" / "games.jsonl"
    out.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    with out.open("w") as f:
        for i in range(n):
            d0, d1 = load_ydk(decks[i % len(decks)]), load_ydk(decks[(i + 1) % len(decks)])
            cfg = DuelConfig(d0, d1, seed=50 + i)
            env = CppEnv(engine, cfg)
            obs, ch, act, tu = env.reset(), [], [], []
            while obs is not None and len(ch) < 400:
                legal = np.flatnonzero(obs["action_mask"])
                a = int(rng.choice(legal))
                ch.append(a)
                act.append(int(env.player))
                tu.append(int(obs["global"][2]))
                obs = env.step(a)
            conf = {"deck0_main": list(d0.main), "deck0_extra": list(d0.extra), "deck1_main": list(d1.main),
                    "deck1_extra": list(d1.extra), "seed": cfg.seed, "hand": cfg.hand, "draw": cfg.draw,
                    "lp": cfg.lp, "setup": [], "shuffle": cfg.shuffle}  # fmt: skip
            f.write(json.dumps({"game_id": f"r{i}", "line_id": f"a-vs-b/r{i}", "decks": ["a", "b"], "config": conf,
                                "auto_single": True, "choices": ch, "actors": act, "turns": tu}) + "\n")  # fmt: skip
    return out


def test_position_score_is_antisymmetric(engine):
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.search.window import position_score

    cfg = gen_search.DuelConfig(*[gen_search.load_ydk(f) for f in sorted((FIXTURES / "decks").glob("*.ydk"))[:2]], seed=3)
    env = CppEnv(engine, cfg)
    obs = env.reset()
    for _ in range(30):
        if obs is None:
            break
        p = int(env.player)
        assert abs(position_score(obs, p, 0) + position_score(obs, p, 1)) < 1e-9
        obs = env.step(0)


def test_window_records_replay_with_legal_targets(engine, tmp_path):
    from ygo_zero.demos.dataset import record_config
    from ygo_zero.engine.actions import Kind
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.search.window import is_window

    src = _random_games(engine, tmp_path)
    out = tmp_path / "window"
    old = sys.argv
    sys.argv = ["gen_window", "--policy", "greedy", "--records", str(src), "--own-frac", "0.2", "--out", str(out)]
    try:
        gen_window.main()
    finally:
        sys.argv = old
    recs = [json.loads(x) for f in out.glob("*.jsonl") for x in f.read_text().splitlines() if x.strip()]
    assert len(recs) == 3
    searched = 0
    for rec in recs:
        assert len(rec["targets"]) == len(rec["labels"]) == len(rec["choices"])
        env = CppEnv(engine, record_config(rec), True)
        obs = env.reset()
        for k, c in enumerate(rec["choices"]):
            t = rec["targets"][k]
            assert (t is None) == (rec["labels"][k] == "skip")
            if t is not None:
                searched += 1
                assert obs["action_mask"][t], "target must be a legal option"
                if rec["labels"][k] == "window":
                    assert is_window(obs, int(env.player))
                    assert int(obs["actions"][t, 0]) in (Kind.PASS, Kind.ACTIVATE)
                else:
                    assert int(obs["global"][5]) == 1 and int(obs["global"][2]) >= 2
            obs = env.step(c)
    assert searched > 0


def test_window_samples_are_targets_with_window_kinds(engine, tmp_path):
    from ygo_zero.demos.dataset import iter_samples

    src = _random_games(engine, tmp_path, n=2)
    out = tmp_path / "window"
    old = sys.argv
    sys.argv = ["gen_window", "--policy", "greedy", "--records", str(src), "--own-frac", "0.2", "--out", str(out)]
    try:
        gen_window.main()
    finally:
        sys.argv = old
    recs = [json.loads(x) for f in out.glob("*.jsonl") for x in f.read_text().splitlines() if x.strip()]
    want = {(r["line_id"], k): t for r in recs for k, t in enumerate(r["targets"]) if t is not None}
    samples = list(iter_samples(engine, out))
    assert len(samples) == len(want) > 0
    for _, a, w, m in samples:
        assert m["kind"] in ("window", "window_own") and w == 1.0
        assert a == want[(m["line_id"], m["k"])]
