"""Per-turn game driver: loop logic with a fake opencode command (no real game is played)."""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


hand_play = _load("hand_play")
play_game = _load("play_game")

_ADVANCE = (
    "python3 -c \"import json,sys; p=sys.argv[1]; st=json.load(open(p)); "
    "st['choices'].append(0); st['labels'].append('x'); st['actors'].append(0); "
    "st['turns'].append(st['turns'][-1] if st['turns'] else 1); "
    "json.dump(st, open(p, 'w'))\" {state}"
)


def _setup(tmp_path, monkeypatch):
    """Isolated WORK + BRIEFS/RUNS and a fresh duel `g` (snake-eye vs tenpai-dragon, seed 11)."""
    monkeypatch.setattr(hand_play, "WORK", tmp_path / "work")
    monkeypatch.setattr(play_game, "BRIEFS", tmp_path / "briefs")
    monkeypatch.setattr(play_game, "RUNS", tmp_path / "briefs" / "runs.tsv")
    monkeypatch.setattr(sys, "argv", ["hand_play.py", "new", "g", "snake-eye", "tenpai-dragon", "11"])
    hand_play.main()
    (tmp_path / "briefs").mkdir(parents=True, exist_ok=True)
    return tmp_path / "briefs"


def _rows(briefs: Path):
    return (briefs / "runs.tsv").read_text().splitlines()


def test_progress_then_max_sessions(tmp_path, monkeypatch):
    briefs = _setup(tmp_path, monkeypatch)
    assert play_game.play("g", "brief.md", "M", 2, _ADVANCE) == "max-sessions"
    rows = _rows(briefs)
    assert len(rows) == 3, rows  # header + 2 sessions
    assert rows[1].split("\t")[0] == "g.t1"
    assert len(hand_play._load("g")["choices"]) == 2  # the fake command advanced the duel
    assert not (briefs / "play-g.pid").exists()


def test_no_progress_twice_stops(tmp_path, monkeypatch):
    briefs = _setup(tmp_path, monkeypatch)
    assert play_game.play("g", "brief.md", "M", 60, "true") == "no-progress"
    assert len(_rows(briefs)) == 3, _rows(briefs)  # header + 2 stagnant sessions
    assert not (briefs / "play-g.pid").exists()


def test_live_pid_refuses_to_start(tmp_path, monkeypatch):
    briefs = _setup(tmp_path, monkeypatch)
    (briefs / "play-g.pid").write_text(str(os.getpid()))
    assert play_game.play("g", "brief.md", "M", 2, "true") == "already-running"
    assert not (briefs / "runs.tsv").exists()


def test_duel_over_runs_final_session(tmp_path, monkeypatch):
    briefs = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(play_game, "duel_status", lambda game: (True, 5, 10))
    assert play_game.play("g", "brief.md", "M", 60, "true") == "over"
    rows = _rows(briefs)
    assert len(rows) == 2, rows
    assert rows[1].split("\t")[0] == "g.final"


def test_transient_error_retries(tmp_path, monkeypatch):
    """A failing command whose log mentions overload is retried (like delegate.sh)."""
    briefs = _setup(tmp_path, monkeypatch)
    script = tmp_path / "flaky.sh"
    script.write_text("#!/bin/sh\necho 'error: overloaded' >&2\nexit 1\n")
    script.chmod(0o755)
    monkeypatch.setattr(play_game.time, "sleep", lambda s: None)
    rc, log = play_game.run_session(game="g", turn=1, brief="b", model="M",
                                    tag="g.t1-retry", cmd_template=f"{script} {{state}}",
                                    tries=2)
    assert rc == 1
    assert log.read_text().count("transient error, retry") == 1
    assert len(_rows(briefs)) == 2


def test_prologue_carries_sheets_every_session(tmp_path, monkeypatch):
    """Every session's prologue carries both decks' compressed sheets + strategy TL;DRs (cached
    per game), under the cap."""
    briefs = _setup(tmp_path, monkeypatch)
    cap = tmp_path / "caps.txt"
    template = f"cat {{prologue}} >> {cap} && echo '===SEP===' >> {cap} && true"
    assert play_game.play("g", "brief.md", "M", 2, template) == "no-progress"
    parts = cap.read_text().split("===SEP===")
    assert len(parts) == 3, parts[-1:]  # two prologues + trailing empty
    first, second, _ = parts
    for part in (first, second):
        assert "== Main" in part  # compressed deck sheets in every session
        assert "## Strategy summaries" in part
        assert len(part) <= play_game.MAX_PROLOGUE_CHARS
    assert "do not re-run log/show unless something changed" in first
    assert (briefs / "play-g.t1.prologue.md").exists()
    assert (briefs / "play-g.sheets.md").exists()  # computed once per game


def test_prologue_reuses_cached_sheets(tmp_path, monkeypatch):
    """build_prologue(game, turn) always includes sheets and reuses the per-game cache."""
    _setup(tmp_path, monkeypatch)
    prologue = play_game.build_prologue("g", 10)
    assert "== Main" in prologue and "## Strategy summaries" in prologue
    cache = tmp_path / "briefs" / "play-g.sheets.md"
    assert cache.exists()
    for line in cache.read_text().splitlines():
        if line.startswith(("== ", "## ", "### ")) or " — " not in line:
            continue
        assert len(line.split(" — ", 1)[1]) <= play_game.SHEET_TEXT_CHARS, line[:80]
    cache.write_text("CACHED")
    assert "CACHED" in play_game.build_prologue("g", 3)  # cache reused, not recomputed
