"""Turn-1 self-imitation selection (scripts/select_turn1_lines.py): best sampled line -> BC record (CPU, tiny)."""

import importlib.util
import json
import sys
from pathlib import Path

from test_turn1_lines import _deck_names, _make_ckpts, _run

from ygo_zero.demos.dataset import iter_samples, sample_kind

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


def _one(deck: str, hand: int, kind: str, rows: list[dict], sample: int = 0) -> dict:
    got = [r for r in rows
           if (r["deck"], r["hand"], r["kind"]) == (deck, hand, kind) and r["sample"] == sample]
    assert len(got) == 1, (deck, hand, kind, sample, len(got))
    return got[0]


def _edited_lines(tmod, tmp_path: Path) -> tuple[Path, str]:
    """Tiny turn1_lines run, hand-edited so deck A hand 0 sample 0 is the known winner.

    Only v_p0 and the (never replay-verified) greedy rows' fields are touched: every sample's
    field keeps its true replayed value, so replay verification passes by construction.
    Deck A hand 1 has a marginal sample (dv 0.05: kept only with a small --min-dv); deck B
    stays train_ok false (holdout/gauntlet stand-in: never written).
    """
    names = _deck_names()
    a = names[0]
    ck = _make_ckpts(tmp_path / "ck")
    line_dir = tmp_path / "lines"
    assert _run(tmod, ck, line_dir) == 0
    rows = _rows(line_dir / "lines.jsonl")
    assert not any(r["stuck"] for r in rows if r["deck"] == a), "tiny lines must not hit the decision cap"
    for r in rows:
        if r["deck"] == a and r["source"] == "random":
            r["train_ok"] = True
    g0, s00, s01 = (_one(a, 0, "greedy", rows, -1), _one(a, 0, "sample", rows, 0), _one(a, 0, "sample", rows, 1))
    g0["v_p0"], g0["field"] = -0.5, s00["field"]  # df 0: qualifies on dv alone
    s00["v_p0"], s01["v_p0"] = 0.5, -1.0  # dv 1.0 beats greedy; the other sample is worse
    g1, s10, s11 = (_one(a, 1, "greedy", rows, -1), _one(a, 1, "sample", rows, 0), _one(a, 1, "sample", rows, 1))
    g1["v_p0"], g1["field"] = 0.0, s10["field"]
    s10["v_p0"], s11["v_p0"] = 0.05, -0.5  # dv 0.05: below the default --min-dv 0.1
    out = tmp_path / "edited.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return out, a


def test_select_best_sample_replays_to_selfimit_kind(engine, tmp_path):
    tmod = _load("turn1_lines")
    lines, a = _edited_lines(tmod, tmp_path)
    mod = _load("select_turn1_lines")
    out = tmp_path / "selfimit"
    assert mod.main(["--lines", str(lines), "--out", str(out)]) == 0
    recs = _rows(out / f"p0-{a}.jsonl")
    assert len(recs) == 1, "only the hand-0 winner beats greedy at the default --min-dv"
    rec = recs[0]
    assert rec["line_id"] == f"{a}/{rec['game_id']}" and rec["game_id"].startswith(f"si-{a}-random-h0-s0")
    assert rec["decks"][0] == a and rec["source"] == "turn1 self-imitation v"
    assert rec["labels"] == ["self" if x == 0 else "opp" for x in rec["actors"]]
    assert all(t == 1 for t in rec["turns"])
    assert (rec["end_field"], rec["end_v"]) != (rec["greedy_field"], rec["greedy_v"])
    assert len(list(out.glob("p0-*.jsonl"))) == 1, "deck B (train_ok false) is never written"
    samples = [(o, act, w, m) for o, act, w, m in iter_samples(engine, out)]
    assert samples, "the selected record must replay through iter_samples"
    assert {m["kind"] for _, _, _, m in samples} == {"selfimit"}
    assert all(w == 1.0 for _, _, w, _ in samples)
    assert all(m["me"] == 0 for _, _, _, m in samples), "only player 0's decisions are samples"
    assert len(samples) == sum(1 for x in rec["actors"] if x == 0)


def test_min_dv_respected_and_no_train_ok_exits_2(tmp_path):
    tmod = _load("turn1_lines")
    lines, a = _edited_lines(tmod, tmp_path)
    mod = _load("select_turn1_lines")
    out = tmp_path / "loose"
    assert mod.main(["--lines", str(lines), "--out", str(out), "--min-dv", "0.01"]) == 0
    assert len(_rows(out / f"p0-{a}.jsonl")) == 2, "dv 0.05 qualifies at --min-dv 0.01"
    assert mod.main(["--lines", str(lines), "--out", str(tmp_path / "strict"), "--min-dv", "2.0"]) == 3
    raw = tmp_path / "lines" / "lines.jsonl"  # unedited run: custom ydk list -> train_ok false everywhere
    assert all(not r["train_ok"] for r in _rows(raw))
    assert mod.main(["--lines", str(raw), "--out", str(tmp_path / "nope")]) == 2


def test_sample_kind_selfimit():
    assert sample_kind("data/demos/selfimit/x/p0-d.jsonl", "self") == "selfimit"
    assert sample_kind("data/demos/selfimit/x/p0-d.jsonl", "opp") == "selfimit"
