import json
import subprocess
import sys
from pathlib import Path

from ygo_zero.demos.dataset import sample_kind

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from relabel_window import confidence, pick

E = {"k": 3, "kind": "window", "cands": [7, 2, 5], "played": 7,
     "scores_all": {"board@h1": [1.0, 3.0, 3.0], "done@h3": [0, 0, 1], "value@h3": [0.1, 0.9, 0.2],
                    "cards@h1": [0, 9, 0]}}


def test_pick_lexicographic_and_ties():
    assert pick(E, ["board@h1"]) == 1  # tie 1 vs 2 -> earlier candidate
    assert pick(E, ["done@h3", "value@h3"]) == 2  # wins within 3 turns beats a higher value
    assert pick(E, ["value@h3"]) == 1


def test_confidence_ignores_cards_and_flat():
    flat = dict(E, scores_all={"board@h1": [1, 1, 1], "value@h3": [0.1, 0.9, 0.2]})
    assert confidence(flat, 1) == 1.0  # only value@h3 separates, and 1 is its top
    assert abs(confidence(E, 2) - 2 / 3) < 1e-9  # board (tie at top incl. 2) + done agree, value does not


def test_sw_kinds():
    assert sample_kind("data/demos/sw-d3/p0-x.jsonl", "window") == "sw-d3"
    assert sample_kind("data/demos/sw-d3/p0-x.jsonl", "own") == "sw-d3_own"
    assert sample_kind("data/demos/window/x.jsonl", "own") == "window_own"


def test_relabel_cli(tmp_path):
    src = tmp_path / "window" / "p0"
    src.mkdir(parents=True)
    rec = {"choices": [0] * 5, "targets": [None, None, None, 7, None], "target_sets": [None, None, None, [7], None],
           "target_conf": [None, None, None, 1.0, None], "labels": ["skip"] * 3 + ["window", "skip"], "search": [E]}
    (src / "a.jsonl").write_text(json.dumps(rec) + "\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/relabel_window.py"), "--src", str(tmp_path / "window/p*"),
                        "--select", "done@h3", "value@h3", "--name", "t", "--out-root", str(tmp_path)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "sw-t" / "p0-a.jsonl").read_text())
    assert out["targets"][3] == 5 and out["target_sets"][3] == [5]
    bad = subprocess.run([sys.executable, str(ROOT / "scripts/relabel_window.py"), "--src", str(tmp_path / "window/p*"),
                          "--select", "nope@h9", "--name", "t", "--out-root", str(tmp_path)], capture_output=True, text=True, check=False)
    assert bad.returncode != 0 and "ALERT" in bad.stderr
