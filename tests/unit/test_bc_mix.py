"""BC data-mix: kind tags, backward-compat shards, --kinds train filtering."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from ygo_zero.demos.dataset import sample_kind
from ygo_zero.model.bc_data import load_shards, split_masks

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sample_kind_from_path_and_label():
    assert sample_kind("data/demos/verified/x.jsonl", "stated") == "line"
    assert sample_kind("data/demos/expanded/x.jsonl", "stated") == "line_expanded"
    assert sample_kind("data/demos/claude/expanded_v2/x.jsonl", "inferred") == "line_expanded"
    assert sample_kind("data/demos/planned-s1/verified/x.jsonl", "stated") == "line"
    assert sample_kind("data/demos/planned-s1/expanded/x.jsonl", "stated") == "line_expanded"
    assert sample_kind("data/demos/planned-s1/improved/verified/x.jsonl", "stated") == "line_improved"
    # expanded wins over improved
    assert sample_kind("data/demos/planned-s1/improved/expanded/x.jsonl", "stated") == "line_expanded"
    assert sample_kind("data/demos/games/g1.jsonl", "interrupt") == "game_interrupt"
    assert sample_kind("data/demos/games/g1.jsonl", "stated") == "game_plan"
    assert sample_kind("data/demos/games/g1.jsonl", "inferred") == "game_plan"


def _fake_data():
    n = 8
    return {
        # lines 0-5 train deck, lines 6-7 holdout/val handling via split_masks below
        "action": np.arange(n),
        "line": np.array([f"l{i // 2}" for i in range(n)]),  # pairs share a base line
        "deck": np.array(["d0"] * 6 + ["d1"] * 2),
        "kind": np.array(["line", "line_improved"] * 4),
    }


def test_kinds_filtering_keeps_val_holdout_whole():
    train_bc = _load("train_bc")
    data = _fake_data()
    masks = split_masks(data, val_frac=0.0, holdout_decks=("d1",))
    assert len(np.flatnonzero(masks["train"])) == 6
    assert len(np.flatnonzero(masks["holdout"])) == 2
    # no filter: everything
    idx = train_bc.select_train_idx(data, masks, None)
    assert len(idx["train"]) == 6 and len(idx["holdout"]) == 2
    # lines only: half the train samples, holdout untouched (still has both kinds)
    idx = train_bc.select_train_idx(data, masks, ["line"])
    assert len(idx["train"]) == 3
    assert {str(data["kind"][i]) for i in idx["train"]} == {"line"}
    assert len(idx["holdout"]) == 2
    assert {str(data["kind"][i]) for i in idx["holdout"]} == {"line", "line_improved"}
    # unknown kind matches nothing
    idx = train_bc.select_train_idx(data, masks, ["game_plan"])
    assert len(idx["train"]) == 0 and len(idx["val"]) == 0 and len(idx["holdout"]) == 2


def test_load_shards_without_kind_gives_unknown(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    np.savez_compressed(
        d / "a.npz", action=np.array([0, 1]), weight=np.array([1.0, 0.5]),
        line=np.array(["l0", "l1"]), deck=np.array(["d0", "d0"]), src=np.array(["g", "g"]),
    )
    data = load_shards(d)
    assert list(data["kind"]) == ["unknown", "unknown"]
    masks = split_masks(data)
    assert set(masks) == {"train", "val", "holdout"}


def test_load_shards_with_kind_roundtrips(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    np.savez_compressed(
        d / "a.npz", action=np.array([0, 1]), weight=np.array([1.0, 1.0]),
        line=np.array(["l0", "l1"]), deck=np.array(["d0", "d0"]),
        kind=np.array(["line", "game_interrupt"]), src=np.array(["g", "g"]),
    )
    data = load_shards(d)
    assert list(data["kind"]) == ["line", "game_interrupt"]


def test_gold_kinds_and_weights():
    from ygo_zero.demos.dataset import GOLD_LABEL_WEIGHT, LABEL_WEIGHT

    assert sample_kind("data/demos/gold/a-vs-b.jsonl", "gold") == "gold"
    assert sample_kind("data/demos/gold/a-vs-b.jsonl", "pass") == "gold_pass"
    # LLM games also label passes "pass": those stay untrained, only gold passes get weight
    assert "pass" not in LABEL_WEIGHT and GOLD_LABEL_WEIGHT["pass"] == 0.5 and "auto" not in GOLD_LABEL_WEIGHT


def test_kind_val_frac_moves_more_gold_lines_to_val():
    lines = np.array([f"g/{i}" for i in range(400)] * 2)
    data = {"action": np.zeros(800, np.int64), "deck": np.array(["d"] * 800), "line": lines,
            "kind": np.array(["gold"] * 400 + ["line"] * 400)}
    base = split_masks(data)
    more = split_masks(data, kind_val_frac={"gold": 0.3})
    g = data["kind"] == "gold"
    assert more["val"][g].sum() > base["val"][g].sum() > 0
    assert (more["val"][~g] == base["val"][~g]).all()  # other kinds untouched
    assert not (base["val"] & ~more["val"]).any()  # val at 0.1 stays val at 0.3


def test_select_score():
    train_bc = _load("train_bc")
    rec = {"train_loss": 0.5, "val": {"acc": 0.8, "per_kind": {"gold": 0.6}}}
    assert train_bc.select_score(rec, "val", -1) == 0.8
    assert train_bc.select_score(rec, "val:gold", -1) == 0.6
    assert train_bc.select_score(rec, "val:gold_pass", -1) == -1
    assert train_bc.select_score(rec, "holdout", -2) == -2
