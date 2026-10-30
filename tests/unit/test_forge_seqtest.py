"""Unit tests for the forge group-sequential paired test (numpy only)."""

import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

from decklight.forge.seqtest import boundaries, deal_order, decide, obf_constant

ROOT = Path(__file__).resolve().parents[2]
SEQTEST = ROOT / "scripts" / "audit" / "forge_seqtest.py"


def _write_arm(d: Path, decks: str, opp: str, seeds, wins, choices):
    """One games-0.jsonl arm: score 1 when `wins`, else 0."""
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "games-0.jsonl", "w") as f:
        for i, s in enumerate(seeds):
            seat = i % 2
            winner = seat if wins[i] else 1 - seat
            f.write(json.dumps({"decks": [decks, opp], "seed": s, "s_seat": seat,
                                "winner": winner, "choices": choices}) + "\n")


def _run(*args):
    return subprocess.run([sys.executable, str(SEQTEST), *args], capture_output=True, text=True, check=False)


def test_obf_constants_and_monotone():
    c6 = obf_constant(6)
    assert 1.98 < c6 < 2.12
    c1 = obf_constant(1)
    assert 1.93 < c1 < 1.99
    b = boundaries(6)
    assert len(b) == 6
    assert all(x > y for x, y in pairwise(b))


def _draw(rng, n, probs):
    vals = np.array([-1.0, 0.0, 1.0])
    return rng.choice(vals, size=n, p=probs)


def test_type_i_error():
    rng = np.random.default_rng(123)
    n_trials, look, K = 4000, 256, 6
    draws = _draw(rng, (n_trials, look * K), [0.13, 0.74, 0.13])
    accepts = sum(
        decide(list(draws[i]), look, K, margin=0.0)["decision"] == "ACCEPT"
        for i in range(n_trials)
    )
    assert accepts / n_trials <= 0.035


def test_power_and_early_stop():
    rng = np.random.default_rng(456)
    n_trials, look, K = 4000, 256, 6
    draws = _draw(rng, (n_trials, look * K), [0.05, 0.75, 0.20])
    looks = []
    accepts = 0
    for i in range(n_trials):
        r = decide(list(draws[i]), look, K, margin=0.03)
        accepts += r["decision"] == "ACCEPT"
        looks.append(r["look"])
    assert accepts / n_trials >= 0.95
    assert float(np.median(looks)) <= 2


def test_futility():
    rng = np.random.default_rng(789)
    n_trials, look, K = 4000, 256, 6
    draws = _draw(rng, (n_trials, look * K), [0.13, 0.74, 0.13])
    looks = []
    rejects = 0
    for i in range(n_trials):
        r = decide(list(draws[i]), look, K, margin=0.03)
        rejects += r["decision"] == "REJECT"
        looks.append(r["look"])
    assert rejects / n_trials >= 0.90
    assert float(np.median(looks)) <= 4


def test_deal_order_and_short():
    keys = [(("B", "A"), 3, 0), (("A", "B"), 1, 1), (("A", "A"), 2, 0)]
    o1 = deal_order(keys, 0)
    assert o1 == deal_order(list(reversed(keys)), 0)
    assert sorted(map(repr, o1)) == sorted(map(repr, keys))
    assert deal_order(keys, 0) == o1
    assert deal_order(keys, 1) != o1 or len(keys) < 2
    r = decide([0.5, -0.5], 256, 6)
    assert r["decision"] == "CONTINUE" and r["look"] == 0 and r["n"] == 2


def test_modest_gain_survives_futility():
    """A real +7 pp pack (above the 3 pp margin) is rarely killed by the non-binding futility stop."""
    rng = np.random.default_rng(321)
    n_trials, look, K = 2000, 256, 6
    draws = _draw(rng, (n_trials, look * K), [0.10, 0.73, 0.17])
    accepts = sum(decide(list(draws[i]), look, K, margin=0.03)["decision"] == "ACCEPT" for i in range(n_trials))
    assert accepts / n_trials >= 0.80


def _hand_diffs(pairs, look, order_seed):
    """Hand-built concatenation: block k's first `look` S - R diffs in deal order."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts" / "audit"))
    import x4b_spec_card
    diffs = []
    for k, (r_dir, s_dir) in enumerate(pairs, start=1):
        rel, spe = x4b_spec_card.games(Path(r_dir)), x4b_spec_card.games(Path(s_dir))
        common = set(rel) & set(spe)
        order = deal_order(list(common), order_seed + k)
        diffs += [spe[c][0] - rel[c][0] for c in order[:look]]
    return diffs


def _arms(tmp_path, wins_r, wins_s, decks="d0", tag=""):
    r, s = tmp_path / f"R{tag}", tmp_path / f"S{tag}"
    n = len(wins_r)
    _write_arm(r, decks, "o", range(n), wins_r, [0, 0])
    _write_arm(s, decks, "o", range(n), wins_s, [1, 1])
    return r, s


def test_multiblock_matches_hand_concatenation(tmp_path):
    """Decision on blocks 1..2 equals decide on the hand-built concatenation."""
    look, K, seed = 8, 6, 0
    n = 12
    r1, s1 = _arms(tmp_path, [False] * n, [True] * n, decks="d1", tag="1")
    r2, s2 = _arms(tmp_path, [True] * (n // 2) + [False] * (n - n // 2),
                   [False] * (n // 2) + [True] * (n - n // 2), decks="d2", tag="2")
    out = tmp_path / "seq.json"
    p = _run("--release", f"{r1},{r2}", "--specialist", f"{s1},{s2}", "--look", str(look),
             "--K", str(K), "--order-seed", str(seed), "--json", str(out))
    assert p.returncode == 0, p.stderr + p.stdout
    got = json.loads(out.read_text())
    assert got["blocks"] == 2
    assert "blocks 2" in p.stdout
    want = decide(_hand_diffs([(r1, s1), (r2, s2)], look, seed), look, K)
    assert got["decision"] == want["decision"] and got["look"] == want["look"]
    assert got["looks"] == want["looks"]


def test_third_block_keeps_looks_1_2(tmp_path):
    """Adding a third block never changes the evaluated looks 1..2."""
    look, K, seed = 8, 6, 3
    n = 12
    # block 1 alternating (CONTINUE), block 2 strong, block 3 hostile
    r1, s1 = _arms(tmp_path, [False] * n, ([True, False] * (n // 2))[:n],
                   decks="d1", tag="1")
    r2, s2 = _arms(tmp_path, [False] * n, [True] * n, decks="d2", tag="2")
    r3, s3 = _arms(tmp_path, [True] * n, [False] * n, decks="d3", tag="3")
    o2, o3 = tmp_path / "s2.json", tmp_path / "s3.json"
    p2 = _run("--release", f"{r1},{r2}", "--specialist", f"{s1},{s2}", "--look", str(look),
              "--K", str(K), "--order-seed", str(seed), "--json", str(o2))
    p3 = _run("--release", f"{r1},{r2},{r3}", "--specialist", f"{s1},{s2},{s3}",
              "--look", str(look), "--K", str(K), "--order-seed", str(seed), "--json", str(o3))
    assert p2.returncode == 0 and p3.returncode == 0, p2.stdout + p3.stdout
    j2, j3 = json.loads(o2.read_text()), json.loads(o3.read_text())
    assert len(j2["looks"]) >= 2
    assert j2["looks"][:2] == j3["looks"][:2]


def test_short_block_exits_2(tmp_path):
    """A block with fewer than L paired games, or < 90% pairing, exits 2."""
    look = 8
    r1, s1 = _arms(tmp_path, [False] * 4, [True] * 4, decks="d1", tag="1")
    p = _run("--release", str(r1), "--specialist", str(s1), "--look", str(look), "--K", "6")
    assert p.returncode == 2 and "ALERT" in p.stdout
    r2, s2 = _arms(tmp_path, [False] * 12, [True] * 12, decks="d2", tag="2")
    # drop most of S2's games so < 90% of R2 pairs
    lines = (s2 / "games-0.jsonl").read_text().splitlines()
    (s2 / "games-0.jsonl").write_text("\n".join(lines[:1]) + "\n")
    p = _run("--release", str(r2), "--specialist", str(s2), "--look", str(look), "--K", "6")
    assert p.returncode == 2 and "ALERT" in p.stdout
