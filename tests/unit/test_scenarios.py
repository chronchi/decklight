"""Scenario puzzles (scripts/scenarios.py): validation, failure reasons, exact replay, turn-2 skip."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "scenarios"


def _load():
    spec = importlib.util.spec_from_file_location("scenarios", ROOT / "scripts" / "scenarios.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scenarios"] = mod
    spec.loader.exec_module(mod)
    return mod


scenarios = _load()


def _fixture() -> dict:
    return json.loads((FIXTURES / "fixture.json").read_text())


def test_fixture_validates(engine):
    from ygo_zero.engine.cpp_env import CppEnv

    names = scenarios.load_names()
    sc = _fixture()
    config, _, errs = scenarios.build_config(sc, seed=0, names=names)
    assert errs == []
    assert config.hand == 0 and len(config.setup) == 4
    assert CppEnv(engine, config).reset() is not None


def test_bad_scenarios_fail_with_reason():
    names = scenarios.load_names()
    base = _fixture()

    unknown = copy.deepcopy(base)
    unknown["p0"]["hand"].append("Not A Real Card XYZ")
    _, _, errs = scenarios.build_config(unknown, names=names)
    assert any("unknown card" in e for e in errs), errs

    foreign = copy.deepcopy(base)
    foreign["p1"]["hand"].append("Blue-Eyes White Dragon")  # not in branded
    _, _, errs = scenarios.build_config(foreign, names=names)
    assert any("card not in deck" in e for e in errs), errs

    copies = copy.deepcopy(base)
    copies["p1"]["hand"].extend(["Ash Blossom & Joyous Spring"] * 3)  # 4 total, branded plays 3
    _, _, errs = scenarios.build_config(copies, names=names)
    assert any("too many copies" in e for e in errs), errs


def test_play_replays_exactly_and_shrinks_decks(engine, tmp_path):
    from ygo_zero.decks import load_ydk
    from ygo_zero.demos.dataset import record_config
    from ygo_zero.engine.cpp_env import CppEnv

    names = scenarios.load_names()
    out = tmp_path / "games"
    scenarios.play_files(
        [FIXTURES / "fixture.json"], out, 2, "random", None, "cpu", 0, engine, names
    )
    recs = [json.loads(x) for x in (out / "fixture-small.jsonl").read_text().splitlines() if x.strip()]
    assert len(recs) == 2
    sc = _fixture()
    full = [load_ydk(ROOT / "decks" / "pool" / f"{sc[f'deck{t}']}.ydk") for t in (0, 1)]
    for rec in recs:
        assert rec["labels"] == ["skip"] * len(rec["choices"])
        assert rec["game_id"].startswith("sc-fixture-small-") and "/sc-" in rec["line_id"]
        assert rec["scenario"] == "fixture-small" and rec["theme"] == "hand_trap_timing"
        assert rec["auto_single"] is True
        cfg = scenarios.build_config(sc, seed=rec["config"]["seed"], names=names)[0]
        assert cfg is not None
        env = CppEnv(engine, record_config(rec), rec["auto_single"])
        obs = env.reset()
        for c in rec["choices"]:
            assert obs is not None
            obs = env.step(c)
        if obs is None:
            assert env.winner == rec["winner"]
        else:
            assert len(rec["choices"]) == scenarios.MAX_DECISIONS and rec["winner"] == 3
        # Placed cards are not drawn from the deck.
        for t in (0, 1):
            placed = [p[1] for p in rec["config"]["setup"] if p[0] == t]
            main = list(full[t].main)
            in_main = 0
            for code in placed:
                if code in main:
                    main.remove(code)
                    in_main += 1
            assert len(rec["config"][f"deck{t}_main"]) == len(full[t].main) - in_main


def test_turn2_first_decision_is_player1_turn2(engine):
    from ygo_zero.engine.cpp_env import CppEnv

    names = scenarios.load_names()
    sc = _fixture()
    sc["start"], sc["focus"] = "turn2", 1
    config, _, errs = scenarios.build_config(sc, seed=0, names=names)
    assert errs == []
    env = CppEnv(engine, config)
    obs = env.reset()
    guard = 0
    while obs is not None and int(obs["global"][2]) < 2 and guard < scenarios.MAX_DECISIONS:
        obs = env.step(scenarios.skip_option(obs))
        guard += 1
    assert obs is not None and int(obs["global"][2]) == 2 and int(env.player) == 1
