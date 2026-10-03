"""Continuous hand-variant expansion: backlog/job choice + expand_lines --remote.

Pure tests on a temp tree, plus expand_lines --remote with remote_verify monkeypatched
(no engine search, no vast, no LLM).
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered to resolve types
    spec.loader.exec_module(mod)
    return mod


dl = _load("datagen_loop")
expand = _load("expand_lines")


def _rec(lid: str, deck: str = "d1") -> dict:
    return {
        "line_id": lid, "deck": deck,
        "config": {
            "deck0_main": [1, 2, 3, 4, 5], "deck0_extra": [],
            "deck1_main": [6, 7, 8], "deck1_extra": [],
            "seed": 0, "hand": 0, "draw": 1, "setup": [[0, 9, 2, 0]],
        },
    }


def _write(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"line_id": i}) + "\n" for i in ids))


def _tree(tmp_path: Path) -> Path:
    demos = tmp_path / "data" / "demos"
    _write(demos / "claude" / "verified" / "d1.jsonl",
           [_rec("d1/l0"), _rec("d1/l1"), _rec("d1/l2")])
    _write_ids(demos / "claude" / "expanded-d1" / "d1.jsonl", ["d1/l0#h0"])
    _write(demos / "planned-s1" / "verified" / "d2.jsonl",
           [_rec("d2/l0", "d2"), _rec("d2/l1", "d2"), _rec("d2/l2", "d2")])
    _write(demos / "planned-s1" / "improved" / "verified" / "d2.jsonl",
           [_rec("d2/l0+imp", "d2"), _rec("d2/l9+imp", "d2")])
    _write_ids(demos / "planned-s1" / "improved" / "expanded-d2" / "d2.jsonl", ["d2/l0#h0"])
    _write_ids(demos / "planned-s1" / "expanded-d2" / "d2.jsonl", ["d2/l9#h0"])
    _write(demos / "planned-s2" / "verified" / "d9.jsonl",
           [_rec(f"d9/l{i}", "d9") for i in range(4)])
    return demos


def test_expander_target_dirs():
    demos = Path("/r/data/demos")
    lines, out, skips = dl.expander_target(demos / "claude" / "verified" / "d1.jsonl")
    assert lines == demos / "claude" / "lines"
    assert out == demos / "claude" / "expanded-d1"
    assert skips == [str(demos / "claude" / "expanded*" / "*.jsonl")]
    lines, out, skips = dl.expander_target(demos / "planned-s1" / "improved" / "verified" / "d2.jsonl")
    assert lines == demos / "planned-s1" / "improved" / "lines"
    assert out == demos / "planned-s1" / "improved" / "expanded-d2"
    assert skips == [str(demos / "planned-s1" / "improved" / "expanded*" / "*.jsonl")]


def test_expander_backlog(tmp_path: Path):
    demos = _tree(tmp_path)
    assert dl.expander_backlog(demos / "claude" / "verified" / "d1.jsonl") == 2
    assert dl.expander_backlog(demos / "planned-s1" / "verified" / "d2.jsonl") == 3  # l9 done, l0-l2 open
    imp = demos / "planned-s1" / "improved" / "verified" / "d2.jsonl"
    # l0+imp is done via the improved expansion; the planned expansion of l9 does NOT count
    assert dl.expander_backlog(imp) == 1
    assert dl.expander_backlog(demos / "missing.jsonl") == 0


def test_choose_expand_job_most_backlog_and_running(tmp_path: Path):
    demos = _tree(tmp_path)
    files = dl.expander_verified_files(tmp_path)
    assert files == sorted(files) and len(files) == 4
    d9 = demos / "planned-s2" / "verified" / "d9.jsonl"
    claude_vf = demos / "claude" / "verified" / "d1.jsonl"
    planned_vf = demos / "planned-s1" / "verified" / "d2.jsonl"
    assert dl.expander_backlog(d9) == 4
    assert dl.choose_expand_job(tmp_path, frozenset()) == d9  # most unexpanded wins
    running = {str(d9.relative_to(tmp_path))}
    assert dl.choose_expand_job(tmp_path, running) == planned_vf  # 3 > claude's 2
    running.add(str(planned_vf.relative_to(tmp_path)))
    assert dl.choose_expand_job(tmp_path, running) == claude_vf
    running.add(str(claude_vf.relative_to(tmp_path)))
    imp = demos / "planned-s1" / "improved" / "verified" / "d2.jsonl"
    assert dl.choose_expand_job(tmp_path, running) == imp  # backlog 1 still counts
    running.add(str(imp.relative_to(tmp_path)))
    assert dl.choose_expand_job(tmp_path, running) is None


def test_expand_command_shape(tmp_path: Path):
    demos = _tree(tmp_path)
    vf = demos / "claude" / "verified" / "d1.jsonl"
    lines, out, skips = dl.expander_target(vf)
    cmd = dl.expand_command(vf, lines, out, skips)
    assert cmd[:5] == ["uv", "run", "--no-sync", "python", "scripts/expand_lines.py"]
    assert "--remote" in cmd
    assert cmd[cmd.index("--variants") + 1] == "8"
    assert cmd[cmd.index("--verified") + 1] == str(vf)
    assert cmd[cmd.index("--lines-dir") + 1] == str(lines)
    assert cmd[cmd.index("--out") + 1] == str(out)
    assert cmd[cmd.index("--skip-done") + 1] == skips[0]
    assert "expander" in dl.CHILD_LIMITS and dl.CHILD_LIMITS["expander"] == 2 * 3600


def _expand_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    vdir, ldir, out = tmp_path / "verified", tmp_path / "lines", tmp_path / "out"
    vdir.mkdir()
    ldir.mkdir()
    (vdir / "d1.jsonl").write_text(json.dumps(_rec("d1/l0")) + "\n")
    (ldir / "d1.jsonl").write_text(json.dumps({"id": "d1/l0", "deck": "d1"}) + "\n")
    return vdir, ldir, out


def _fake_verdicts():
    from ygo_zero.decks import Deck
    from ygo_zero.engine.duel import DuelConfig, Placement

    cfg = DuelConfig(Deck((1, 2), (), (), "d0"), Deck((4,), (), (), "d1"),
                     seed=0, setup=(Placement(0, 9, 2, 0),))
    return SimpleNamespace(ok=True, reason="ok", config=cfg, choices=[0], labels=["stated"],
                           skipped_steps=[], executed_steps=1, inferred_frac=0.0)


def test_expand_remote_uses_remote_verify(tmp_path: Path, monkeypatch, capsys):
    vdir, ldir, out = _expand_tree(tmp_path)
    seen: list = []

    def fake_remote(items, timeout, jobs=None, session="lines"):
        seen.append((items, timeout, session))
        return [_fake_verdicts() for _ in items]

    monkeypatch.setattr(expand, "remote_verify", fake_remote)
    argv = ["expand_lines", "--verified", str(vdir), "--lines-dir", str(ldir),
            "--variants", "2", "--seed", "0", "--out", str(out), "--remote"]
    monkeypatch.setattr(sys, "argv", argv)
    expand.main()
    assert len(seen) == 1  # one call for the whole batch
    items, timeout, session = seen[0]
    assert timeout == expand.VARIANT_TIMEOUT and session == "lines"
    assert len(items) == 2 and all(len(it) == 7 for it in items)  # verify-item tuples
    assert all(it[0]["id"].startswith("d1/l0#h") for it in items)
    assert all(isinstance(it[1], tuple) and isinstance(it[2], tuple) and it[3] == () for it in items)
    assert all(it[5] == 0 for it in items)  # seed carried over
    recs = [json.loads(x) for x in (out / "d1.jsonl").read_text().splitlines()]
    assert len(recs) == 2 and all(r["variant_of"] == "d1/l0" for r in recs)
    assert "2/2 hand variants verify" in capsys.readouterr().out


def test_expand_local_fallback_without_remote(tmp_path: Path, monkeypatch):
    vdir, ldir, out = _expand_tree(tmp_path)
    used = []

    def fake_map(fn, items, jobs, initializer=None):
        used.append(items)
        return [SimpleNamespace(ok=False, reason="fake") for _ in items]

    def no_remote(*a, **k):
        raise AssertionError("remote_verify must not be called without --remote")

    monkeypatch.setattr(expand, "map_ordered", fake_map)
    monkeypatch.setattr(expand, "remote_verify", no_remote)
    argv = ["expand_lines", "--verified", str(vdir), "--lines-dir", str(ldir),
            "--variants", "2", "--seed", "0", "--out", str(out), "--jobs", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    expand.main()
    assert len(used) == 1 and len(used[0]) == 2
    assert all(len(it) == 5 for it in used[0])  # local 5-tuples unchanged
