"""Unit tests for the forge round prep + release card (no torch, no games)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "scripts" / "forge_round_prep.py"
RELEASE_CARD = ROOT / "scripts" / "audit" / "forge_release_card.py"


def _run(*args):
    return subprocess.run([sys.executable, *[str(a) for a in args]], capture_output=True, text=True, check=False)


def _pack(d: Path, archetype: str, train, heldout=()):
    d.mkdir(parents=True, exist_ok=True)
    (d / "pack.json").write_text(json.dumps({
        "archetype": archetype,
        "decks": [{"stem": s, "role": "train"} for s in train]
        + [{"stem": s, "role": "heldout"} for s in heldout]}))
    return d


def _write_arm(d: Path, wins, choices):
    """One games-0.jsonl arm over shared keys; score 1 when `wins`."""
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "games-0.jsonl", "w") as f:
        for i, w in enumerate(wins):
            seat = i % 2
            f.write(json.dumps({"decks": ["d0", "o0"], "seed": i, "s_seat": seat,
                                "winner": seat if w else 1 - seat, "choices": choices}) + "\n")


def test_pairs_excludes_archetype(tmp_path):
    pack = _pack(tmp_path / "p", "testarch", ["t1", "t2"], ["h1"])
    opps = tmp_path / "opps.txt"
    opps.write_text("keep1\ntestarch-x\nkeep2\n")
    out = tmp_path / "x.pairs"
    p = _run(PREP, "pairs", "--pack", pack, "--opponents", opps, "--out", out)
    assert p.returncode == 0, p.stderr
    assert "pairs 6" in p.stdout
    lines = out.read_text().splitlines()
    assert sorted(lines) == sorted(f"{d}:{o}" for d in ("t1", "t2", "h1") for o in ("keep1", "keep2"))
    opps.write_text("testarch-a\ntestarch-b\n")
    p = _run(PREP, "pairs", "--pack", pack, "--opponents", opps, "--out", out)
    assert p.returncode == 2


def test_anchor_keeps_train_drops_rest(tmp_path):
    pack = _pack(tmp_path / "p", "testarch", ["abc-dragon"])
    for s in ("elfnote-f59", "elfnote-f61"):  # real lists, unused by the subcommand itself
        shutil.copy(ROOT / "decks" / "forge" / "f0" / "train" / "elfnote" / f"{s}.ydk", pack / f"{s}.ydk")
    out = tmp_path / "anchor"
    p = _run(PREP, "anchor", "--packs", pack, "--out", out)
    assert p.returncode == 0, p.stderr + p.stdout
    names = {q.stem for q in out.glob("*.ydk")}
    assert "invoked-meta1" in names  # x4b train list kept
    assert "abc-dragon" not in names  # pack stem dropped
    assert "elfnote-f59" not in names  # forge stem dropped
    assert "invoked-v3" not in names  # x4b-scored (heldout) stem dropped
    assert len(names) >= 100


def test_klmap_train_only_and_dup(tmp_path):
    a = _pack(tmp_path / "a", "aa", ["t1", "t2"], ["h1"])
    out = tmp_path / "kl.json"
    p = _run(PREP, "klmap", "--packs", a, "--default", "rel.pt", "--out", out)
    assert p.returncode == 0, p.stderr
    m = json.loads(out.read_text())
    assert m == {"default": "rel.pt", "decks": {"t1": f"{a}/policy/weights.safetensors", "t2": f"{a}/policy/weights.safetensors"}}
    b = _pack(tmp_path / "b", "bb", ["t2"], ["h2"])
    p = _run(PREP, "klmap", "--packs", a, b, "--default", "rel.pt", "--out", out)
    assert p.returncode == 2


def test_seeds_rounding(tmp_path):
    pairs = tmp_path / "x.pairs"
    pairs.write_text("".join(f"d{i}:o\n" for i in range(10)))
    assert _run(PREP, "seeds", "--pairs", pairs, "--games", 1536).stdout.strip() == "77"
    assert _run(PREP, "seeds", "--pairs", pairs, "--games", 1520).stdout.strip() == "76"
    pairs.write_text("")
    assert _run(PREP, "seeds", "--pairs", pairs, "--games", 1536).returncode == 2


def _fake_card(root: Path, n: int = 20, drop=()):
    """Card where N - R is +100 pp on `new` and 0 elsewhere."""
    card = root / "card"
    _write_arm(card / "R-new", [False] * n, [0, 0])
    _write_arm(card / "N-new", [True] * n, [1, 1])
    for tag in ("old0", "res", "ga"):
        _write_arm(card / f"R-{tag}", [True] * 10, [0, 0])
        _write_arm(card / f"N-{tag}", [True] * 10, [1, 1])
    for d in drop:
        shutil.rmtree(card / d)
    (root / "verify.json").write_text(json.dumps({"new": {
        "pack_id": "new-x", "decision": "ACCEPT", "look": 1, "n": 280,
        "mean_pp": 50.0, "rci_pp": [40.0, 60.0]}}))
    return card


def test_release_card_gates(tmp_path):
    card = _fake_card(tmp_path)
    p = _run(RELEASE_CARD, card, "--verify", tmp_path / "verify.json",
             "--new", "new", "--old", "old0", "--name", "r1")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "GATE kept PASS" in p.stdout and "GATE old PASS" in p.stdout
    assert "GATE reserve PASS" in p.stdout and "GATE gauntlet PASS" in p.stdout
    assert "RELEASE r1 yes" in p.stdout
    res = json.loads((tmp_path / "release_card.json").read_text())
    assert res["release"] is True and res["new"]["new"]["kept"] == 2.0
    assert res["pooled_kept"] == 2.0


def test_release_card_missing_reserve(tmp_path):
    card = _fake_card(tmp_path, drop=["N-res"])
    p = _run(RELEASE_CARD, card, "--verify", tmp_path / "verify.json",
             "--new", "new", "--old", "old0", "--name", "r1")
    assert "ALERT" in p.stdout and "GATE reserve FAIL" in p.stdout
    assert "RELEASE r1 no" in p.stdout
    assert p.returncode == 0  # N-res missing fails the gate; R-res present so no exit 2
    card2 = _fake_card(tmp_path / "d2", drop=["R-res", "N-res"])
    p = _run(RELEASE_CARD, card2, "--verify", tmp_path / "d2" / "verify.json",
             "--new", "new", "--old", "old0", "--name", "r1")
    assert p.returncode == 2
