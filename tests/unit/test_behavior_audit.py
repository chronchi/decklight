"""Behaviour-audit scripts (scripts/audit/): pairing, aggregation, parsing, and one real replay.

LLM calls are mocked (or never reach the network: judge --dry-run renders engine-only).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

AUDIT = Path(__file__).resolve().parents[2] / "scripts" / "audit"
CARD_TEST = Path(__file__).resolve().parents[2] / "runs" / "ft-curve-20261022-1928" / "card-test"

sys.path.insert(0, str(AUDIT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_mod", AUDIT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sm():
    return _load("sm_common")


def _rec(decks, seed, seat, winner, idx):
    return {"decks": list(decks), "seed": seed, "s_seat": seat, "game_idx": idx,
            "winner": winner, "truncated": False, "turns": 3,
            "choices": [0, 1], "actors": [seat, 1 - seat]}


def _write_arm(path: Path, recs: list[dict]):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "games-0.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_flip_pairing_counts_three_deals(tmp_path):
    mod = _load("flip_audit")
    decks = ["BlueEyes", "Hero"]
    a = [_rec(decks, 1, 0, 1, 0),  # A loss
         _rec(decks, 2, 0, 0, 1),  # A win
         _rec(decks, 3, 1, 1, 2)]  # A win (seat 1)
    b = [_rec(decks, 1, 0, 0, 0),  # B win  -> L->W
         _rec(decks, 2, 0, 1, 1),  # B loss -> W->L
         _rec(decks, 3, 1, 1, 2)]  # B win  -> W->W
    _write_arm(tmp_path / "a", a)
    _write_arm(tmp_path / "b", b)
    pairs, a_only, b_only = mod.pair_arms(a, b)
    assert len(pairs) == 3 and not a_only and not b_only
    counts, keys = mod.flip_counts(pairs)
    assert counts == {"L->W": 1, "W->L": 1, "W->W": 1}
    assert len(keys["L->W"]) == 1 and len(keys["W->L"]) == 1


def test_judge_aggregate_and_verdict_parsing():
    mod = _load("behavior_judge")
    props = [{"id": "b1", "title": "T1"}, {"id": "b2", "title": "T2"}]
    verdicts = {("b1", "base", k): (k % 2 == 0) for k in range(10)}
    verdicts.update({("b2", "base", k): True for k in range(10)})
    wins = {("base", k): 1.0 if k < 5 else 0.0 for k in range(10)}
    prev = mod.aggregate(props, verdicts, wins)
    assert prev["b1"]["arms"]["base"]["prev"] == pytest.approx(0.5)
    assert prev["b1"]["arms"]["base"]["n_with"] == 5
    s = prev["b1"]["arms"]["base"]
    assert 0.0 < s["lo"] < s["prev"] < s["hi"] < 1.0
    # even keys (with) are k=0..8 -> wins 1,1,1,0,0 = 0.6; without k=1..9 -> 1,1,0,0,0 = 0.4
    assert s["win_with"] == pytest.approx(0.6) and s["win_without"] == pytest.approx(0.4)
    assert prev["b2"]["arms"]["base"]["prev"] == pytest.approx(1.0)
    assert prev["b2"]["arms"]["base"]["win_without"] is None
    fenced = '```json\n{"b1": true, "b2": false}\n```'
    assert mod.parse_verdicts(fenced, ["b1", "b2"]) == {"b1": True, "b2": False}
    assert mod.parse_verdicts("no json here", ["b1"]) is None
    assert mod.parse_verdicts('{"b1": true}', ["b1", "b2"]) is None  # missing id
    ids = ["b01-fire_pass", "b02-chip_attack"]
    assert mod.parse_verdicts('{"b01": true, "b02": false}', ids) == {ids[0]: True, ids[1]: False}  # short ids
    assert mod.parse_verdicts('{"b01": true, "b02": false}', ["b01-x", "b01-y"]) is None  # ambiguous prefix


def test_proposer_parsing_tolerant_of_fences():
    mod = _load("behavior_proposer")
    body = ('Here are the behaviours:\n```json\n[{"id": "x", "title": "Sets starter", '
            '"description": "d", "evidence": ["g1", "g2"], "detector_hint": "h"}]\n```')
    got = mod.parse_behaviors(body)
    assert len(got) == 1 and got[0]["id"] == "x" and got[0]["evidence"] == ["g1", "g2"]
    assert mod.parse_behaviors("nothing parseable") == []
    assert mod.parse_behaviors('{"behaviours": []}') == []


def test_proposer_main_with_mocked_llm(tmp_path, monkeypatch):
    mod = _load("behavior_proposer")
    import decklight.demos.llm as _llm

    (tmp_path / "pairs").mkdir()
    (tmp_path / "pairs" / "L-to-W-000.txt").write_text("pair one text")
    (tmp_path / "pairs" / "W-to-L-000.txt").write_text("pair two text")
    cand = [{"id": "holds-maxx-c", "title": "Holds Maxx C", "description": "d",
             "evidence": ["L-to-W-000", "W-to-L-000"], "detector_hint": "h"}]

    def fake_many(prompts, **k):
        assert len(prompts) == 1  # 2 pairs, batch default 4 -> 1 batch
        return ['```json\n' + json.dumps(cand) + '\n```']

    monkeypatch.setattr(_llm, "complete_many", fake_many)
    monkeypatch.setattr(_llm, "complete", lambda *a, **k: json.dumps(cand))
    assert mod.main(["--pairs", str(tmp_path / "pairs"), "--out", str(tmp_path / "b.json")]) == 0
    out = json.loads((tmp_path / "b.json").read_text())
    assert len(out) == 1 and out[0]["evidence"] == ["L-to-W-000", "W-to-L-000"]
    assert out[0]["id"].startswith("b01-")


def test_play_features_summarize_and_top_changes():
    mod = _load("play_features")

    def game(mix, t1, sig, score=1.0):
        return {"ok": True, "mix": mix, "n_ours": sum(mix.values()), "t1": t1, "t2": [],
                "first_n": len(sig), "sig": tuple(sig), "score": score}

    base = [game({"summon": 2}, ["Blue-Eyes White Dragon"], [("SUMMON", "Blue-Eyes White Dragon")]) for _ in range(4)]
    bewd = ("SUMMON", "Blue-Eyes White Dragon")
    ft = [game({"summon": 2}, ["Blue-Eyes White Dragon"], [bewd]) for _ in range(2)]
    ft += [game({"activate": 1}, ["Blue-Eyes White Dragon", "Maxx C"], [bewd, ("ACTIVATE", "Maxx C")])
           for _ in range(2)]
    stats = {"base": mod.summarize(base), "final": mod.summarize(ft)}
    assert stats["base"]["n"] == 4 and stats["base"]["diversity"] == pytest.approx(0.25)
    assert stats["base"]["top_line"]["share"] == pytest.approx(1.0)
    assert stats["final"]["diversity"] == pytest.approx(0.5)
    top = mod.top_changes(stats, "base", "turn1")
    assert top[0]["card"] == "Maxx C" and top[0]["max_delta"] == pytest.approx(0.5)
    assert stats["final"]["mix"]["activate"]["share"] == pytest.approx(0.333)
    assert stats["final"]["mix"]["summon"]["share"] == pytest.approx(0.667)


def test_render_sm_real_record_reproduces_winner(tmp_path, sm):
    by_seat = {}
    for path in sorted((CARD_TEST / "base").glob("games-*.jsonl")):  # proc shards hold one seat; scan files
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by_seat.setdefault(r["s_seat"], r)
        if set(by_seat) == {0, 1}:
            break
    assert set(by_seat) == {0, 1}, "need one game per seat"
    recs = [by_seat[0], by_seat[1]]
    engine, names = sm.Engine(), sm.CardNames()
    for rec in recs:
        text, ok = sm.render_record(rec, engine, names)
        assert ok, f"game {rec.get('game_idx')} replay mismatch"
        assert ">>" in text and text.startswith(f"game {rec.get('game_idx')}")
    mod = _load("render_sm")
    src = tmp_path / "games-0.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert mod.main([str(src), "--out", str(tmp_path / "rendered")]) == 0
    assert len(list((tmp_path / "rendered").glob("game-*.txt"))) == 2
