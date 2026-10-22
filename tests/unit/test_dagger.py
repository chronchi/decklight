"""DAgger records (scripts/gen_dagger.py): stub games replay, targets label student decisions, samples train them."""

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


gen_dagger = _load("gen_dagger")


def _generate(tmp_path: Path, games: int = 6) -> Path:
    out = tmp_path / "dagger"
    argv = ["gen_dagger", "--out", str(out), "--games", str(games),
            "--decks", str(FIXTURES / "decks"), "--policy", "stub",
            "--seed", "7", "--threads", "2", "--beta", "0.3"]
    old = sys.argv
    sys.argv = argv
    try:
        gen_dagger.main()
    finally:
        sys.argv = old
    return out


def _records(out: Path) -> list[dict]:
    files = sorted(out.glob("*.jsonl"))
    assert files, "no dagger files written"
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]


def _by_k(recs: list[dict]) -> list[dict]:
    return sorted(recs, key=lambda r: int(r["game_id"].split("-")[-1]))


def test_dagger_stub_games_replay(engine, tmp_path):
    from decklight.demos.dataset import record_config
    from decklight.engine.cpp_env import CppEnv, VecEnv

    recs = _records(_generate(tmp_path))
    assert len(recs) == 6
    ids = [r["game_id"] for r in recs]
    assert len(set(ids)) == len(ids), "game_ids must be unique"
    for seat, rec in zip([0, 1] * 3, _by_k(recs), strict=True):
        assert rec["student_seat"] == seat, f"{rec['game_id']}: student seat must alternate 0, 1, ..."
    for rec in recs:
        n = len(rec["choices"])
        assert len(rec["targets"]) == len(rec["labels"]) == len(rec["actors"]) == len(rec["turns"]) == n
        assert isinstance(rec["beta"], bool) and rec["student_seat"] in (0, 1)
        assert rec["level"] == "L4" and rec["notes"] == [] and rec["auto_single"] is True
        assert "dagger" in rec["source"]
        for k in range(n):
            student = rec["actors"][k] == rec["student_seat"]
            assert (rec["targets"][k] is not None) == student, f"{rec['game_id']} k={k}"
            if student:
                assert rec["labels"][k] in ("gold", "pass")
            else:
                assert rec["labels"][k] == "opp"
        env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
        obs = env.reset()
        for c in rec["choices"]:
            obs = env.step(c)
        if rec["winner"] == VecEnv.TRUNCATED:
            continue  # cut games are written; their states are still useful
        assert obs is None, f"{rec['game_id']}: replay did not end the duel"
        assert env.winner == rec["winner"], rec["game_id"]
        assert env.turn == rec["end_turn"], rec["game_id"]


def test_dagger_samples_train_targets(engine, tmp_path):
    from decklight.demos.dataset import iter_samples, record_config, sample_kind
    from decklight.engine.cpp_env import CppEnv

    assert sample_kind("data/demos/dagger/x.jsonl", "pass") == "dagger_pass"
    assert sample_kind("data/demos/dagger/x.jsonl", "gold") == "dagger"
    out = _generate(tmp_path)
    recs = _records(out)
    by_line = {r["line_id"]: r for r in recs}
    # expected count: student decisions whose target is a legal option there
    want = 0
    for rec in recs:
        env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
        obs = env.reset()
        for k, c in enumerate(rec["choices"]):
            if rec["actors"][k] == rec["student_seat"]:
                t = rec["targets"][k]
                assert t is not None
                if 0 <= int(t) < obs["action_mask"].shape[0] and bool(obs["action_mask"][int(t)]):
                    want += 1
            obs = env.step(c)
    got = list(iter_samples(engine, out))
    kinds = {m["kind"] for _, _, _, m in got}
    assert kinds <= {"dagger", "dagger_pass"}, kinds
    assert len(got) == want, f"{len(got)} samples != {want} legal student decisions"
    assert got, "no dagger samples yielded"
    for _, action, weight, meta in got:
        rec = by_line[meta["line_id"]]
        assert action == rec["targets"][meta["k"]], f"{meta['line_id']} k={meta['k']}"
        assert weight > 0


def test_samples_without_targets_yield_choices(engine, tmp_path):
    from decklight.demos.dataset import iter_samples

    out = _generate(tmp_path)
    rec = _records(out)[0]
    assert rec["targets"] is not None
    rec = {k: v for k, v in rec.items() if k != "targets"}  # a pre-dagger record
    legacy = out / "legacy.jsonl"  # under dagger/ so gold weights apply
    legacy.write_text(json.dumps(rec) + "\n")
    got = [(a, m) for _, a, _, m in iter_samples(engine, legacy)]
    assert got, "legacy record yielded nothing"
    for a, m in got:
        assert a == rec["choices"][m["k"]], f"k={m['k']}: without targets the choice is the target"
