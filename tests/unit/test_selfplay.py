"""League self-play records (scripts/gen_selfplay.py): stub games replay, league schedule, holdout, window input."""

import importlib.util
import json
import shutil
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


gen_selfplay = _load("gen_selfplay")

LEAGUE = "self:0.5,greedy:0.5"


def _generate(tmp_path: Path, games: int = 6, seed: int = 7, league: str = LEAGUE,
              decks: Path | None = None, holdout: Path | None = None) -> Path:
    out = tmp_path / f"sp-{seed}-{games}"
    argv = ["gen_selfplay", "--out", str(out), "--games", str(games),
            "--decks", str(decks or (FIXTURES / "decks")),
            "--ygoagent-decks", str(FIXTURES / "decks"),
            "--policy", "stub", "--league", league, "--seed", str(seed), "--threads", "2"]
    if holdout is not None:
        argv += ["--holdout-file", str(holdout)]
    old = sys.argv
    sys.argv = argv
    try:
        gen_selfplay.main()
    finally:
        sys.argv = old
    return out


def _records(out: Path) -> list[dict]:
    files = sorted(out.glob("*.jsonl"))
    assert files, "no selfplay files written"
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]


def test_selfplay_stub_games_replay(engine, tmp_path):
    from ygo_zero.demos.dataset import record_config
    from ygo_zero.engine.cpp_env import CppEnv, VecEnv

    out = _generate(tmp_path)
    recs = _records(out)
    assert len(recs) == 6
    ids = [r["game_id"] for r in recs]
    assert len(set(ids)) == len(ids), "game_ids must be unique"
    assert [r["game_id"] for r in sorted(recs, key=lambda r: r["game_id"])] == sorted(ids)
    for r in recs:
        assert r["game_id"].startswith("sp-7-")
        assert r["line_id"].endswith(f"/{r['game_id']}")
        assert r["level"] == "L4" and r["notes"] == [] and r["auto_single"] is True
        assert r["opponent"] in ("self", "greedy") and r["temp"] == 1.0
        assert "selfplay" in r["source"]
    for seat, rec in zip([0, 1] * 3, sorted(recs, key=lambda r: int(r["game_id"].split("-")[-1])), strict=True):
        assert rec["student_seat"] == seat, f"{rec['game_id']}: student seat must alternate 0, 1, ..."
    by_file = {f.stem: [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
               for f in sorted(out.glob("*.jsonl"))}
    for stem, rs in by_file.items():
        for r in rs:  # one file per student deck
            assert r["line_id"].startswith(f"{stem}-vs-"), f"{r['line_id']} in {stem}.jsonl"
    for rec in recs:
        n = len(rec["choices"])
        assert len(rec["actors"]) == len(rec["turns"]) == len(rec["labels"]) == n
        for k in range(n):
            assert (rec["labels"][k] == "self") == (rec["actors"][k] == rec["student_seat"])
        env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
        obs = env.reset()
        for c in rec["choices"]:
            obs = env.step(c)
        if rec["winner"] == VecEnv.TRUNCATED:
            continue  # cut games are written; their states are still useful
        assert obs is None, f"{rec['game_id']}: replay did not end the duel"
        assert env.winner == rec["winner"], rec["game_id"]
        assert env.turn == rec["end_turn"], rec["game_id"]


def test_league_draw_follows_weights(tmp_path):
    a = _records(_generate(tmp_path / "a", games=20, seed=11, league="self:0.5,greedy:0.5,random:0.0"))
    b = _records(_generate(tmp_path / "b", games=20, seed=11, league="self:0.5,greedy:0.5,random:0.0"))
    opps_a = [r["opponent"] for r in sorted(a, key=lambda r: r["game_id"])]
    opps_b = [r["opponent"] for r in sorted(b, key=lambda r: r["game_id"])]
    assert opps_a == opps_b, "same seed must give the same league schedule"
    assert set(opps_a) == {"self", "greedy"}, f"both weighted entries should appear: {set(opps_a)}"
    assert "random" not in opps_a, "a weight-0 entry must never appear"


def test_holdout_stems_never_drawn(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    for f in (FIXTURES / "decks").glob("*.ydk"):
        shutil.copy(f, pool / f.name)
    holdout = tmp_path / "HOLDOUT"
    holdout.write_text("BlueEyes\n")
    recs = _records(_generate(tmp_path / "o", games=8, league="self:0.5,greedy:0.5",
                              decks=pool, holdout=holdout))
    assert len(recs) == 8
    for r in recs:
        assert "BlueEyes" not in r["decks"], r["line_id"]
        assert not r["line_id"].startswith("BlueEyes-vs-"), r["line_id"]
    assert not (tmp_path / "o" / "BlueEyes.jsonl").exists()


def test_records_are_valid_window_input(engine, tmp_path):
    gen_window = _load("gen_window")
    out = _generate(tmp_path / "w", games=6)
    rec_files = sorted(str(f) for f in out.glob("*.jsonl"))
    wout = tmp_path / "window"
    argv = ["gen_window", "--policy", "greedy", "--records", *rec_files,
            "--games", "0", "--width", "2", "--k", "1", "--horizon", "1",
            "--seed", "0", "--out", str(wout)]
    old = sys.argv
    sys.argv = argv
    try:
        gen_window.main()
    finally:
        sys.argv = old
    got = [json.loads(line) for f in sorted(wout.glob("*.jsonl"))
           for line in f.read_text().splitlines() if line.strip()]
    assert len(got) == 6, f"window search should emit one record per input game, got {len(got)}"
