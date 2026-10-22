"""Teacher records (scripts/gen_teacher.py): random-policy games replay and map to teacher kinds."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen_teacher = _load("gen_teacher")


def _generate(tmp_path: Path, games: int = 6) -> Path:
    out = tmp_path / "teacher"
    argv = ["gen_teacher", "--out", str(out), "--games", str(games),
            "--decks", str(FIXTURES / "decks"), "--policy", "random",
            "--seed", "7", "--threads", "2", "--sample-frac", "0.5"]
    old = sys.argv
    sys.argv = argv
    try:
        gen_teacher.main()
    finally:
        sys.argv = old
    return out


def _records(out: Path) -> list[dict]:
    files = sorted(out.glob("*.jsonl"))
    assert files, "no teacher files written"
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]


def test_teacher_random_games_replay(engine, tmp_path):
    from decklight.demos.dataset import record_config
    from decklight.engine.cpp_env import CppEnv

    recs = _records(_generate(tmp_path))
    assert len(recs) == 6
    ids = [r["game_id"] for r in recs]
    assert len(set(ids)) == len(ids), "game_ids must be unique"
    for rec in recs:
        assert len(rec["labels"]) == len(rec["choices"])
        assert len(rec["actors"]) == len(rec["choices"])
        assert len(rec["turns"]) == len(rec["choices"])
        assert set(rec["labels"]) <= {"gold", "pass"}
        assert rec["level"] == "L4" and rec["notes"] == [] and rec["auto_single"] is True
        env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
        obs = env.reset()
        for c in rec["choices"]:
            obs = env.step(c)
        assert obs is None, f"{rec['game_id']}: replay did not end the duel"
        assert env.winner == rec["winner"], rec["game_id"]
        assert env.turn == rec["end_turn"], rec["game_id"]


def test_teacher_samples_kinds_only(engine, tmp_path):
    from decklight.demos.dataset import iter_samples, sample_kind

    assert sample_kind("data/demos/teacher/x.jsonl", "pass") == "teacher_pass"
    assert sample_kind("data/demos/teacher/x.jsonl", "gold") == "teacher"
    kinds = {m["kind"] for _, _, _, m in iter_samples(engine, _generate(tmp_path))}
    assert kinds == {"teacher", "teacher_pass"}, kinds
