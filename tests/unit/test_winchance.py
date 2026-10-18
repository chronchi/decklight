"""Win-chance calibration: PAV, the per-bucket map, clipping, and the fit script."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ygo_zero.eval import winchance as W


def test_pav_monotone_on_random_data():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = rng.random(2000)
        y = (rng.random(2000) < x).astype(float)
        ux, f = W.pav_increasing(x, y)
        assert len(ux) > 10  # non-trivial fit
        assert bool((np.diff(ux) > 0).all()), "xs must be sorted unique"
        assert bool((np.diff(f) >= 0).all()), "fit must be non-decreasing"


def test_pav_pools_ties():
    ux, f = W.pav_increasing(np.array([0.5, 0.5, 0.5]), np.array([0.0, 1.0, 1.0]))
    assert len(ux) == 1 and ux[0] == 0.5
    assert f[0] == pytest.approx(2 / 3)


def test_map_applied_per_bucket():
    m = W.CalibMap([([0.0, 1.0], [0.1, 0.2]), ([0.0, 1.0], [0.7, 0.8]),
                    ([0.0, 1.0], [0.5, 0.6])])
    assert W.bucket_of(1) == 0 and W.bucket_of(2) == 0
    assert W.bucket_of(3) == 1 and W.bucket_of(4) == 1
    assert W.bucket_of(5) == 2 and W.bucket_of(99) == 2
    # Same raw value maps differently per bucket (v=-1 -> p_raw=0).
    assert m.p(-1.0, 2) == pytest.approx(0.1)
    assert m.p(-1.0, 3) == pytest.approx(0.7)
    assert m.p(-1.0, 4) == pytest.approx(0.7)
    assert m.p(-1.0, 9) == pytest.approx(0.5)


def test_clip_never_0_or_100():
    m = W.CalibMap([([0.0, 1.0], [0.0, 1.0])] * 3)
    for v in (-1.0, -0.999, 0.999, 1.0, -5.0, 5.0):
        for turn in (1, 3, 4, 10):
            p = m.p(v, turn)
            assert W.P_LO <= p <= W.P_HI, (v, turn, p)
            assert p not in (0.0, 1.0)


def test_load_calib_roundtrip(tmp_path):
    m = W.CalibMap([([0.0, 0.5, 1.0], [0.1, 0.4, 0.9])] * 3)
    p = tmp_path / "calib.json"
    p.write_text(json.dumps({"map": m.to_dict()}))
    m2 = W.load_calib(p)
    assert m2.p(-1.0, 1) == pytest.approx(0.1)
    assert m2.p(1.0, 5) == pytest.approx(0.9)


def _synthetic_game(i: int, rng) -> tuple[dict, dict]:
    """One game record + one value line (actor-view values, y = actor won)."""
    decks = ["DeckA", "DeckB"]
    seed = 1000 + i
    winner = int(rng.integers(0, 2))
    n = int(rng.integers(40, 80))
    actors = [int(rng.integers(0, 2)) for _ in range(n)]
    turns = [1 + (k * 8 // n) for k in range(n)]  # turns 1..8, all buckets hit
    v = []
    for a in actors:
        edge = 0.3 if a == winner else -0.3
        v.append(round(float(np.clip(rng.normal(edge, 0.5), -1, 1)), 4))
    rec = {"game_idx": i, "seed": seed, "decks": decks, "winner": winner,
           "truncated": False, "choices": [0] * n}
    val = {"game_idx": i, "seed": seed, "decks": decks, "s_seat": 0,
           "k": list(range(n)), "actor": actors, "turn": turns, "v": v}
    return rec, val


def test_fit_script_on_synthetic_jsonl(tmp_path, monkeypatch):
    from fit_winchance import main

    rng = np.random.default_rng(7)
    arm = "hppo-p2"
    games_d = tmp_path / "src" / arm
    score_d = tmp_path / "score" / arm
    games_d.mkdir(parents=True)
    score_d.mkdir(parents=True)
    recs, vals = zip(*(_synthetic_game(i, rng) for i in range(1000)))
    (games_d / "games-0.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (score_d / "values-0.jsonl").write_text("\n".join(json.dumps(g) for g in vals) + "\n")
    out = tmp_path / "winchance_calib.json"
    report = tmp_path / "winchance-calibration.md"
    monkeypatch.setattr(sys, "argv", ["fit_winchance.py", "--src", str(tmp_path / "src"),
                                      "--score-dir", str(tmp_path / "score"),
                                      "--arms", arm, "--out", str(out),
                                      "--report", str(report), "--seed", "0"])
    assert main() == 0
    d = json.loads(out.read_text())
    assert d["n_games_fit"] + d["n_games_held"] == 1000 or d["n_games"] == 1000
    assert len(d["map"]["knots"]) == 3
    for e in d["map"]["knots"]:
        assert 0 < len(e["x"]) <= 50 and len(e["x"]) == len(e["y"])
        assert all(b >= a for a, b in zip(e["y"], e["y"][1:])), "map must be monotone"
    for m in d["metrics"]:
        assert m["n_held_mean"] >= 1000
        assert m["status"] == "fitted", m
    assert d["metrics"][0]["ece_cal_mean"] < d["metrics"][0]["ece_raw_mean"]
    text = report.read_text()
    assert "ECE raw" in text and "Reliability" in text
    # Report is written fresh: a second run must not duplicate it.
    assert main() == 0
    text2 = report.read_text()
    assert text2.count("# Win-chance calibration") == 1


def test_fit_script_identity_fallback_on_worse_bucket(tmp_path, monkeypatch):
    """Strict gate: a bucket worse after calibration gets an identity map + exit 2."""
    import fit_winchance

    from ygo_zero.eval import winchance as W

    rng = np.random.default_rng(7)
    arm = "hppo-p2"
    games_d = tmp_path / "src" / arm
    score_d = tmp_path / "score" / arm
    games_d.mkdir(parents=True)
    score_d.mkdir(parents=True)
    recs, vals = zip(*(_synthetic_game(i, rng) for i in range(500)))
    (games_d / "games-0.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (score_d / "values-0.jsonl").write_text("\n".join(json.dumps(g) for g in vals) + "\n")
    out = tmp_path / "c.json"
    # Invert the calibration (calibrated = 1 - raw): held-out ECE/Brier must get worse.
    real_apply = W.apply_knots
    monkeypatch.setattr(W, "apply_knots",
                        lambda p, xs, ys: np.clip(1.0 - np.asarray(p, dtype=float), W.P_LO, W.P_HI))
    monkeypatch.setattr(sys, "argv", ["fit_winchance.py", "--src", str(tmp_path / "src"),
                                      "--score-dir", str(tmp_path / "score"),
                                      "--arms", arm, "--out", str(out),
                                      "--report", str(tmp_path / "r.md"), "--seed", "0"])
    with pytest.raises(SystemExit) as e:
        fit_winchance.main()
    assert e.value.code != 0
    d = json.loads(out.read_text())
    assert any(m["status"].startswith("IDENTITY") for m in d["metrics"])
    for m, k in zip(d["metrics"], d["map"]["knots"]):
        if m["status"].startswith("IDENTITY"):
            assert k["x"] == [0.0, 1.0] and k["y"] == [0.0, 1.0]
    # The identity map really is raw-through-clip.
    assert real_apply(1.0, [0.0, 1.0], [0.0, 1.0]) == pytest.approx(W.P_HI)
    assert real_apply(0.0, [0.0, 1.0], [0.0, 1.0]) == pytest.approx(W.P_LO)


def test_fit_script_rejects_small_buckets(tmp_path, monkeypatch):
    """Too little data: the n >= 1000 gate must fail loudly (exit 2)."""
    from fit_winchance import main

    rng = np.random.default_rng(11)
    arm = "hppo-p"
    games_d = tmp_path / "src" / arm
    score_d = tmp_path / "score" / arm
    games_d.mkdir(parents=True)
    score_d.mkdir(parents=True)
    recs, vals = zip(*(_synthetic_game(i, rng) for i in range(10)))
    (games_d / "games-0.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (score_d / "values-0.jsonl").write_text("\n".join(json.dumps(g) for g in vals) + "\n")
    monkeypatch.setattr(sys, "argv", ["fit_winchance.py", "--src", str(tmp_path / "src"),
                                      "--score-dir", str(tmp_path / "score"),
                                      "--arms", arm, "--out", str(tmp_path / "c.json"),
                                      "--report", str(tmp_path / "r.md"), "--seed", "0"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code != 0
