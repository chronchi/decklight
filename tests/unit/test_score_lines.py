"""Score turn-1 lines by playing them out (scripts/score_lines.py + --by rollout selection).

CPU, tiny d=16 random policy (tests/unit/test_turn1_lines.py fixtures); < 1 min.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from test_turn1_lines import _deck_names, _make_ckpts, _run

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _scored_lines(tmp_path: Path, samples: int = 3) -> tuple[Path, dict, Path]:
    """Tiny turn1_lines run scored with the tiny policy (procs=1: no spawn)."""
    tmod = _load("turn1_lines")
    ck = _make_ckpts(tmp_path / "ck")
    line_dir = tmp_path / "lines"
    assert _run(tmod, ck, line_dir) == 0
    mod = _load("score_lines")
    out = tmp_path / "scored.jsonl"
    rc = mod.main(["--ckpt", str(ck["ckpt"]), "--emb", str(ck["emb"]),
                   "--lines", str(line_dir / "lines.jsonl"), "--out", str(out),
                   "--cands", "3", "--samples", str(samples), "--horizon", "2",
                   "--max-rollout", "200", "--procs", "1", "--threads", "1",
                   "--device", "cpu", "--batch", "16", "--width", "4"])
    assert rc == 0, f"score_lines exited {rc}"
    return out, ck, line_dir / "lines.jsonl"


def test_candidates_get_finite_rollout_scores(engine, tmp_path):
    mod = _load("score_lines")
    out, _, _ = _scored_lines(tmp_path)
    rows = _rows(out)
    assert rows, "scored groups must write rows"
    kinds = {(r["deck"], r["source"], r["hand"]) for r in rows}
    assert kinds, "at least one group scored"
    for r in rows:
        assert np.isfinite(r["rollout"]), r
        assert len(r["rollout_samples"]) == 3
        assert all(np.isfinite(s) for s in r["rollout_samples"])
        assert np.isfinite(r["rollout_done"])
        assert r["rollout"] == float(sum(r["rollout_samples"]) / len(r["rollout_samples"]))
        assert isinstance(r["rollout_capped"], int) and r["rollout_capped"] >= 0
    summary = json.loads(Path(str(out) + ".summary.json").read_text())
    assert summary["groups"] >= 1 and summary["candidates"] == len(rows)
    assert summary["capped_share"] is not None and summary["capped_share"] <= 0.10
    # branch_start check helper: the greedy line's replay reaches turn 2 (or the duel ended)
    for r in rows:
        if r["kind"] == "greedy":
            assert mod.branch_turn(engine, r) in (2, None)


def test_common_random_numbers_identical_on_rescore(tmp_path):
    mod = _load("score_lines")
    out, ck, lines = _scored_lines(tmp_path)
    out2 = tmp_path / "scored2.jsonl"
    rc = mod.main(["--ckpt", str(ck["ckpt"]), "--emb", str(ck["emb"]),
                   "--lines", str(lines), "--out", str(out2),
                   "--cands", "3", "--samples", "3", "--horizon", "2",
                   "--max-rollout", "200", "--procs", "1", "--threads", "1",
                   "--device", "cpu", "--batch", "16", "--width", "4"])
    assert rc == 0
    key = lambda r: (r["deck"], r["source"], r["hand"], r["kind"], r["sample"])  # noqa: E731
    a = {key(r): (r["rollout"], list(r["rollout_samples"])) for r in _rows(out)}
    b = {key(r): (r["rollout"], list(r["rollout_samples"])) for r in _rows(out2)}
    assert set(a) == set(b) and a, "same candidates both runs"
    for k in a:
        assert a[k][0] == b[k][0] and a[k][1] == b[k][1], f"CRN violated for {k}"


def test_select_by_rollout_beats_greedy_plus_min_dr(engine, tmp_path):
    _load("turn1_lines")
    mod = _load("score_lines")
    out, _, _ = _scored_lines(tmp_path)
    sel = _load("select_turn1_lines")
    rows = _rows(out)
    names = _deck_names()
    a = names[0]
    for r in rows:  # deck A trainable; others stay train_ok false (never written)
        if r["deck"] == a and r["source"] == "random":
            r["train_ok"] = True
    by_hand: dict[int, list[dict]] = {}
    for r in rows:
        if r["deck"] == a and r["source"] == "random":
            by_hand.setdefault(r["hand"], []).append(r)
    for h, g in by_hand.items():  # only hand 0 sample 0 beats greedy (by 1.0)
        gr = [r for r in g if r["kind"] == "greedy"][0]
        for s in g:
            if s["kind"] == "sample":
                s["rollout"] = float(gr["rollout"] + (1.0 if (h == 0 and s["sample"] == 0) else -5.0))
    edited = tmp_path / "scored_edited.jsonl"
    edited.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    keep = tmp_path / "keep"
    assert sel.main(["--lines", str(edited), "--out", str(keep), "--by", "rollout"]) == 0
    recs = _rows(keep / f"p0-{a}.jsonl")
    assert len(recs) == 1 and recs[0]["game_id"].startswith(f"si-{a}-random-h0-s0")
    assert recs[0]["source"] == "turn1 self-imitation rollout"
    g0 = [r for r in rows if (r["deck"], r["hand"], r["kind"]) == (a, 0, "greedy")][0]
    assert recs[0]["rollout"] == g0["rollout"] + 1.0 and recs[0]["greedy_rollout"] == g0["rollout"]
    assert len(list(keep.glob("p0-*.jsonl"))) == 1, "deck B (train_ok false) is never written"
    # a strict --min-dr drops the marginal winner; unscored rows are ignored
    assert sel.main(["--lines", str(edited), "--out", str(tmp_path / "strict"),
                     "--by", "rollout", "--min-dr", "2.0"]) == 3
    noroll = tmp_path / "noroll.jsonl"
    noroll.write_text("\n".join(json.dumps({k: v for k, v in r.items() if not k.startswith("rollout")})
                                for r in rows) + "\n")
    assert sel.main(["--lines", str(noroll), "--out", str(tmp_path / "none"),
                     "--by", "rollout"]) in (2, 3)
