"""Stage M play-out anchor (scripts/turn_anchor.py): both arms play out from the same state and return S samples each."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

import turn_anchor as ta  # noqa: E402
from gen_search import greedy_logits  # noqa: E402
from test_window import _random_games  # noqa: E402


def test_anchor_arms(engine, tmp_path):
    import json

    from ygo_zero.search.window import WindowSearch

    rec = json.loads((_random_games(engine, tmp_path, n=1, max_len=80)).read_text().splitlines()[0])
    k = 40
    rec["labels"] = ["prefix"] * k + ["stated"] * (len(rec["choices"]) - k)
    assert ta.turn_start(rec) == k
    ws = WindowSearch(engine, greedy_logits, horizon=1, samples=3, rollout_temp=2.0)
    out = dict(ws.run_many([(0, ta.anchor_gen(ws, rec, 3, 7))], 4))
    sm, sp = out[0]
    assert len(sm) == 3 and len(sp) == 3
    assert all(isinstance(x, float) for x in sm + sp)


def test_mean_var():
    assert ta.mean_var([1.0, 3.0]) == (2.0, 2.0)
    with pytest.raises(ZeroDivisionError):
        ta.mean_var([])
