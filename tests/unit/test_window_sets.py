"""Window target sets + confidence (brief multiscore): dataset alt/weight and shard equiv."""

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


def _random_games(engine, tmp_path: Path, n: int = 2) -> Path:
    from decklight.decks import load_ydk
    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import DuelConfig

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


def _window_out(engine, tmp_path, args):
    src = _random_games(engine, tmp_path)
    out = tmp_path / "window"
    old = sys.argv
    sys.argv = ["gen_window", "--policy", "greedy", "--records", str(src), "--out", str(out)] + args
    try:
        gen_window.main()
    finally:
        sys.argv = old
    return out


def test_target_sets_flow_to_samples_and_shards(engine, tmp_path):
    from decklight.demos.dataset import iter_samples
    from decklight.model.bc_data import _shard_one, equivalent_mask

    out = _window_out(engine, tmp_path, ["--own-frac", "0.2", "--horizons", "1", "3", "--tie-eps", "0.5"])
    recs = [json.loads(x) for f in out.glob("*.jsonl") for x in f.read_text().splitlines() if x.strip()]
    assert recs and all("target_sets" in r and "target_conf" in r for r in recs)
    assert any(s is not None for r in recs for s in r["target_sets"])
    samples = list(iter_samples(engine, out))
    assert samples
    with_alt = [(o, a, w, m) for o, a, w, m in samples if "alt" in m]
    assert with_alt
    for obs, a, w, m in with_alt:
        mask = obs["action_mask"].astype(bool)
        assert m["alt"] and all(mask[x] for x in m["alt"])
        assert a in m["alt"]  # the set always contains the best (the target)
        k = m["k"]
        rec = next(r for r in recs if r["line_id"] == m["line_id"])
        assert rec["target_sets"][k] is not None
        assert rec["targets"][k] == a
        assert abs(w - 1.0 * float(rec["target_conf"][k])) < 1e-6 or rec["target_conf"][k] is None
    # Shard: equiv = equivalent_mask(target) OR legal alt options.
    import decklight.model.bc_data as bcd
    from decklight.model.bc_data import _ENGINE as _BE

    bcd._ENGINE = engine
    try:
        files = sorted(out.glob("*.jsonl"))
        shard = tmp_path / "s.npz"
        _shard_one((str(files[0]), str(shard), "window"))
    finally:
        bcd._ENGINE = _BE
    z = dict(np.load(shard))
    assert "equiv" in z and "weight" in z
    # Re-derive the first sample's expectation from iter_samples order.
    first = None
    for obs, a, w, m in iter_samples(engine, files[0]):
        first = (obs, a, w, m)
        break
    assert first is not None
    obs, a, w, m = first
    want = equivalent_mask(obs["actions"], a, obs.get("action_mask")).copy()
    for b in m.get("alt", []):
        if bool(obs["action_mask"][b]):
            want[b] = True
    assert (z["equiv"][0] == want).all()
    assert abs(float(z["weight"][0]) - w) < 1e-6
