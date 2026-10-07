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
    conf = {(r["line_id"], k): c for r in recs for k, c in enumerate(r.get("target_conf") or []) if c is not None}
    samples = list(iter_samples(engine, out))
    assert len(samples) == len(want) > 0
    for _, a, w, m in samples:
        assert m["kind"] in ("window", "window_own")
        assert a == want[(m["line_id"], m["k"])]
        # weight is the confidence (base 1.0 x target_conf); records without conf keep 1.0
        c = conf.get((m["line_id"], m["k"]))
        assert w == (c if c is not None else 1.0)


def test_value_scoring_constant_value_keeps_pass(engine, tmp_path):
    """With a learned value that is 0 everywhere, every unfinished branch ties, so a window keeps PASS (ranked
    first) unless a branch ends the duel; scores stay within [-1, 1]."""
    from ygo_zero.demos.dataset import record_config
    from ygo_zero.engine.duel import Engine
    from ygo_zero.search.window import WindowSearch

    src = _random_games(engine, tmp_path, n=2)
    calls = []

    def zero_value(obs):
        calls.append(len(obs["global"]))
        return np.zeros(len(obs["global"]), np.float32)

    ws = WindowSearch(Engine(), gen_search.greedy_logits, own_frac=0.2, value=zero_value)
    recs = [json.loads(x) for x in src.read_text().splitlines()]
    jobs = ((i, ws.game_gen(r, record_config(r), i)) for i, r in enumerate(recs))
    n = 0
    for _, res in ws.run_many(jobs, 4):
        for e in res.log:
            n += 1
            assert all(-1.0 <= s <= 1.0 for s in e["scores"])
            if e["kind"] == "window" and max(e["scores"]) == 0.0:
                assert not e["fired"]
    assert n > 0 and calls


def _run_search(engine, recs, tmp_path, **kw):
    from ygo_zero.demos.dataset import record_config
    from ygo_zero.engine.duel import Engine
    from ygo_zero.search.window import WindowSearch

    kw.setdefault("own_frac", 0.2)
    ws = WindowSearch(Engine(), gen_search.greedy_logits, **kw)
    jobs = ((i, ws.game_gen(r, record_config(r), i)) for i, r in enumerate(recs))
    return {i: res for i, res in ws.run_many(jobs, 4)}, ws


def test_defaults_match_explicit_equivalent(engine, tmp_path):
    import json as _json

    src = _random_games(engine, tmp_path, n=2)
    recs = [_json.loads(x) for x in src.read_text().splitlines()]
    dflt, _ = _run_search(engine, recs, tmp_path)
    xplt, _ = _run_search(engine, recs, tmp_path, horizons=(1,), samples=1, select="board@h1")
    assert set(dflt) == set(xplt)
    for i in dflt:
        assert dflt[i].targets == xplt[i].targets
        assert dflt[i].sets == xplt[i].sets
        assert dflt[i].conf == xplt[i].conf
        assert len(dflt[i].log) == len(xplt[i].log)
        for a, b in zip(dflt[i].log, xplt[i].log):
            assert a["cands"] == b["cands"] and a["scores"] == b["scores"]
            assert a["scores_all"] == b["scores_all"] and a["set"] == b["set"]
            assert a["conf"] == b["conf"]
    n_log = sum(len(res.log) for res in dflt.values())
    assert n_log > 0
    for res in dflt.values():
        for e in res.log:
            # tie_eps=0: the target alone (exact ties keep the candidate order: PASS first at a window)
            assert e["set"] == [e["chosen"]]
            assert 0.0 <= e["conf"] <= 1.0


def test_multi_horizon_scores(engine, tmp_path):
    import json as _json

    src = _random_games(engine, tmp_path, n=2)
    recs = [_json.loads(x) for x in src.read_text().splitlines()]
    out, _ = _run_search(engine, recs, tmp_path, horizons=(1, 3))
    n = 0
    for res in out.values():
        for e in res.log:
            n += 1
            sa = e["scores_all"]
            for kk in ("board@h1", "board@h3", "done@h1", "done@h3"):
                assert kk in sa
                assert len(sa[kk]) == len(e["cands"])
            assert any(k.startswith("cards@h") for k in sa)
            for v in sa["done@h1"] + sa["done@h3"]:
                assert v in (-1.0, 0.0, 1.0)
            for a, b in zip(sa["done@h1"], sa["done@h3"]):
                if a in (-1.0, 1.0):  # duel ended by h1: terminal persists
                    assert b == a
    assert n > 0


def test_samples_reproducible(engine, tmp_path):
    import json as _json

    src = _random_games(engine, tmp_path, n=1)
    recs = [_json.loads(x) for x in src.read_text().splitlines()]
    r1, _ = _run_search(engine, recs, tmp_path, horizons=(1,), samples=3)
    r2, _ = _run_search(engine, recs, tmp_path, horizons=(1,), samples=3)
    for i in r1:
        assert [e["scores_all"] for e in r1[i].log] == [e["scores_all"] for e in r2[i].log]
        assert [e["chosen"] for e in r1[i].log] == [e["chosen"] for e in r2[i].log]
    assert sum(len(r.log) for r in r1.values()) > 0


def test_tie_eps_large_gives_full_set_conf_one(engine, tmp_path):
    import json as _json

    src = _random_games(engine, tmp_path, n=1)
    recs = [_json.loads(x) for x in src.read_text().splitlines()]
    out, _ = _run_search(engine, recs, tmp_path, horizons=(1,), tie_eps=1e9)
    n = 0
    for res in out.values():
        for e in res.log:
            n += 1
            assert sorted(e["set"]) == sorted(e["cands"])
            assert e["conf"] == 1.0
        for k, s in res.sets.items():
            assert sorted(s) == sorted(res.log[[e["k"] for e in res.log].index(k)]["cands"])
            assert res.conf[k] == 1.0
    assert n > 0


def test_value_multi_horizon_keys(engine, tmp_path):
    import json as _json

    import numpy as np

    src = _random_games(engine, tmp_path, n=1)

    def stub_value(obs):
        n = len(obs["global"])
        return np.linspace(-0.5, 0.5, n).astype(np.float32)

    recs = [_json.loads(x) for x in src.read_text().splitlines()]
    out, _ = _run_search(engine, recs, tmp_path, horizons=(1, 3), value=stub_value, select="value@h1")
    n = 0
    for res in out.values():
        for e in res.log:
            n += 1
            assert "value@h1" in e["scores_all"] and "value@h3" in e["scores_all"]
            assert all(-1.0 <= s <= 1.0 for s in e["scores"])
    assert n > 0


def test_game_gen_honest_false_is_bit_identical(engine, tmp_path):
    import json as _json

    from ygo_zero.demos.dataset import record_config
    from ygo_zero.engine.duel import Engine
    from ygo_zero.search.window import WindowSearch

    src = _random_games(engine, tmp_path, n=1)
    recs = [_json.loads(x) for x in src.read_text().splitlines()]
    rec, cfg = recs[0], record_config(recs[0])
    ws = WindowSearch(Engine(), gen_search.greedy_logits, own_frac=0.2, seed=0)
    dflt = dict(ws.run_many([(0, ws.game_gen(rec, cfg, 0))], 4))[0]
    ws2 = WindowSearch(Engine(), gen_search.greedy_logits, own_frac=0.2, seed=0)
    expl = dict(ws2.run_many([(0, ws2.game_gen(rec, cfg, 0, honest=False))], 4))[0]
    assert expl.targets == dflt.targets
    assert expl.kinds == dflt.kinds and expl.sets == dflt.sets and expl.conf == dflt.conf
    assert len(expl.log) == len(dflt.log)
    for a, b in zip(expl.log, dflt.log):
        assert a == b
    assert expl.det_proposals == 0 and expl.det_skipped == 0
    assert sum(len(r.log) for r in (dflt,)) > 0


def test_game_gen_honest_skip_never_uses_true_config(engine, tmp_path):
    import json as _json

    from ygo_zero.demos.dataset import record_config
    from ygo_zero.engine.duel import Engine
    from ygo_zero.search.window import WindowSearch

    src = _random_games(engine, tmp_path, n=2)
    recs = [_json.loads(x) for x in src.read_text().splitlines()]
    for r in recs:  # short prefixes: early turn>=2 own-turn decisions still search, determinizer replays stay cheap
        r["choices"] = r["choices"][:100]
    ws = WindowSearch(Engine(), gen_search.greedy_logits, own_frac=1.0, seed=0, samples=1)
    orig = ws.rollouts_gen
    seen: list = []
    cfgs: list = []

    def spy(config, auto_single, branches, me, **kw):
        w = kw.get("worlds")
        if w is not None:
            seen.append(list(w))
            cfgs.append(config)
        return (yield from orig(config, auto_single, branches, me, **kw))

    ws.rollouts_gen = spy  # type: ignore[method-assign]
    cfgs_rec = [record_config(r) for r in recs]
    jobs = ((i, ws.game_gen(r, cfgs_rec[i], i, honest=True, det_tries=8, det_fallback="skip"))
            for i, r in enumerate(recs))
    out = dict(ws.run_many(jobs, 4))
    assert all(r.replay_ok for r in out.values())
    assert sum(r.det_proposals for r in out.values()) > 0
    assert sum(r.det_accepted + r.det_fallbacks for r in out.values()) > 0
    assert seen, "honest search ran no rollout with worlds"
    for worlds, cfg in zip(seen, cfgs, strict=True):
        assert worlds, "skip mode must pass only accepted worlds"
        for w in worlds:
            assert isinstance(w, tuple), f"skip mode must not use the true config, got {type(w)}"
            wcfg, _wprefix = w
            assert wcfg is not cfg, "rollout used the true config (None-substitute) in skip mode"
