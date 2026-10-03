"""Overnight data-generation driver: pure helpers + expand_lines --skip-done.

Pure functions only, on a temp dir. No vast, no LLM, no engine-heavy runs
(the --skip-done test monkeypatches the verify worker).
"""

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    pool = tmp_path / "decks" / "pool"
    pool.mkdir(parents=True)
    for d in ("blue-eyes", "sky-striker", "branded", "mikanko", "kewl-tune"):
        (pool / f"{d}.ydk").write_text("#ydk\n")
    (pool / "HOLDOUT").write_text("mikanko\n")

    demos = tmp_path / "data" / "demos"
    (demos / "verified").mkdir(parents=True)
    (demos / "verified" / "blue-eyes.jsonl").write_text(
        '{"line_id": "a"}\n{"line_id": "b"}\n\n{"line_id": "c"}\n'
    )
    (demos / "verified" / "sky-striker.jsonl").write_text('{"line_id": "a"}\n')
    (demos / "verified" / "kewl-tune.jsonl").write_text(
        "".join(f'{{"line_id": "k{i}"}}\n' for i in range(5))
    )
    (demos / "planned-s1" / "verified").mkdir(parents=True)
    (demos / "planned-s1" / "verified" / "blue-eyes.jsonl").write_text(
        '{"line_id": "d"}\n{"line_id": "e"}\n'
    )
    (demos / "planned-s1" / "verified" / "blue-eyes+guide.jsonl").write_text('{"line_id": "f"}\n')
    (demos / "planned-s1" / "report").mkdir(parents=True)
    (demos / "planned-s1" / "report" / "blue-eyes.json").write_text("[]")
    (demos / "planned-s3" / "report").mkdir(parents=True)
    (demos / "planned-s3" / "report" / "blue-eyes.json").write_text("[]")

    games = demos / "games"
    games.mkdir(parents=True)
    (games / "blue-eyes-vs-branded.jsonl").write_text(
        '{"decks": ["blue-eyes", "branded"]}\nnot json\n\n'
        '{"decks": ["sky-striker", "blue-eyes"]}\n'
    )
    return tmp_path


def test_choose_planner_deck_fewest_verified(repo: Path):
    counts = dl.verified_counts(repo)
    assert counts["blue-eyes"] == 6  # 3 in verified + 2 in planned-s1 + 1 in +guide
    assert counts["sky-striker"] == 1
    assert dl.choose_planner_deck(repo, set()) == "branded"  # 0-way tie with mikanko, alphabetical
    assert dl.choose_planner_deck(repo, {"branded"}) == "mikanko"
    assert dl.choose_planner_deck(repo, {"branded", "mikanko"}) == "sky-striker"


def test_next_planner_seed(repo: Path):
    assert dl.next_planner_seed(repo, "blue-eyes") == 4  # planned-s1 and planned-s3 exist
    assert dl.next_planner_seed(repo, "sky-striker") == 1


def test_game_decks_exclude_holdout_and_kewl_tune(repo: Path):
    assert dl.eligible_game_decks(repo) == ["blue-eyes", "branded", "sky-striker"]
    counts = dl.count_games_per_deck(repo)
    assert counts == {"blue-eyes": 2, "branded": 1, "sky-striker": 1}  # malformed/blank ignored
    w = dl.game_weights(["blue-eyes", "branded", "sky-striker"], counts)
    assert w == {"blue-eyes": pytest.approx(1 / 3), "branded": 0.5, "sky-striker": 0.5}
    assert dl.game_weights(["new-deck"], {}) == {"new-deck": 1.0}


def test_pick_game_pair(repo: Path):
    decks = dl.eligible_game_decks(repo)
    counts = dl.count_games_per_deck(repo)
    rng = random.Random(0)
    pairs = {dl.pick_game_pair(decks, counts, rng) for _ in range(200)}
    assert pairs  # non-empty
    for a, b in pairs:
        assert a != b and a in decks and b in decks
    firsts = {a for a, _ in pairs}
    assert len(firsts) > 1  # first player is random
    with pytest.raises(ValueError):
        dl.pick_game_pair(["solo"], {}, random.Random(0))


def test_flat_name_mapping():
    flat = "data_demos_planned-s1_expanded-blue-eyes"
    assert dl.flat_to_path(flat) == Path("data/demos/planned-s1/expanded-blue-eyes")
    assert dl.path_to_flat("data/demos/planned-s1/expanded-blue-eyes") == flat


def test_parse_train_first_line():
    line = ("{'train': 49450, 'val': 8083, 'holdout': 5067} decks: "
            "['blue-eyes', 'branded', 'kewl-tune']")
    counts, decks = dl.parse_train_first_line(line)
    assert counts == {"train": 49450, "val": 8083, "holdout": 5067}
    assert decks == ["blue-eyes", "branded", "kewl-tune"]


def test_metrics_best(tmp_path: Path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(
        '{"epoch": 0, "val": {"acc": 0.76}, "holdout": {"acc": 0.71}}\n'
        '{"epoch": 1, "val": {"acc": 0.81}, "holdout": {"acc": 0.73}}\n'
        '{"epoch": 2, "val": {"acc": 0.80}}\n'  # holdout may be absent (empty split)
        "not json\n"
    )
    assert dl.metrics_best(p) == {
        "val_acc": 0.81, "val_epoch": 1, "holdout_acc": 0.73, "holdout_epoch": 1,
    }
    assert dl.metrics_best(tmp_path / "missing.jsonl") == {
        "val_acc": None, "val_epoch": None, "holdout_acc": None, "holdout_epoch": None,
    }


def test_planner_command_variant():
    full = dl.planner_command("blue-eyes", 3, "low")
    assert "--remote" in full and "--variant" in full and "low" in full
    assert full[full.index("--out-dir") + 1] == "data/demos/planned-s3"
    assert "--variant" not in dl.planner_command("blue-eyes", 1, None)


def _write_skip_fake(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    vdir = tmp_path / "verified"
    ldir = tmp_path / "lines"
    out = tmp_path / "out"
    vdir.mkdir()
    ldir.mkdir()

    def rec(lid: str) -> dict:
        return {
            "line_id": lid, "deck": "d1",
            "config": {
                "deck0_main": [1, 2, 3, 4, 5], "deck0_extra": [],
                "deck1_main": [6, 7, 8], "deck1_extra": [],
                "seed": 0, "hand": 0, "draw": 1, "setup": [[0, 9, 2, 0]],
            },
        }

    (vdir / "d1.jsonl").write_text(json.dumps(rec("d1/l0")) + "\n" + json.dumps(rec("d1/l1")) + "\n")
    (ldir / "d1.jsonl").write_text(
        json.dumps({"id": "d1/l0", "deck": "d1"}) + "\n" + json.dumps({"id": "d1/l1", "deck": "d1"}) + "\n"
    )
    done = tmp_path / "already.jsonl"
    done.write_text('{"line_id": "d1/l0#h0"}\n{"line_id": "d1/l0#h1"}\n{"line_id": "other#h2"}\n')
    return vdir, ldir, out, done


def test_expand_collect_done_ids(tmp_path: Path):
    _, _, _, done = _write_skip_fake(tmp_path)
    assert expand.collect_done_ids([str(done)]) == {"d1/l0", "other"}
    assert expand.base_id_of("d1/l0#h3") == "d1/l0"


def test_expand_skip_done(tmp_path: Path, monkeypatch, capsys):
    vdir, ldir, out, done = _write_skip_fake(tmp_path)
    seen: list = []

    def fake_map(fn, items, jobs, initializer=None):
        seen.extend(items)
        return [SimpleNamespace(ok=False, reason="fake") for _ in items]

    monkeypatch.setattr(expand, "map_ordered", fake_map)
    argv = ["expand_lines", "--verified", str(vdir), "--lines-dir", str(ldir),
            "--variants", "2", "--seed", "0", "--out", str(out), "--jobs", "1",
            "--skip-done", str(done)]
    monkeypatch.setattr(sys, "argv", argv)
    expand.main()
    ids = {it[0]["id"] for it in seen}
    assert ids and all(i.startswith("d1/l1#") for i in ids)  # d1/l0's variants were skipped
    assert "skipped 1 records" in capsys.readouterr().out

    seen.clear()
    monkeypatch.setattr(sys, "argv", argv[:-2])  # no --skip-done: nothing skipped
    expand.main()
    ids = {it[0]["id"] for it in seen}
    assert {i.split("#")[0] for i in ids} == {"d1/l0", "d1/l1"}
