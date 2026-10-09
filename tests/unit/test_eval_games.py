"""Eval games saved by rl.evaluate.save_games pair and replay in scripts/audit/rl_ga_compare.py (engine only)."""

import importlib.util
import json
from pathlib import Path

from ygo_zero.decks import load_ydk
from ygo_zero.eval.match import play_match
from ygo_zero.eval.policies import GreedyPolicy, RandomPolicy
from ygo_zero.rl.evaluate import save_games

ROOT = Path(__file__).resolve().parents[2]


def _compare_mod():
    p = ROOT / "scripts" / "audit" / "rl_ga_compare.py"
    spec = importlib.util.spec_from_file_location("rl_ga_compare_t", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_saved_eval_games_pair_and_replay(engine, tmp_path):
    names = ["decks/pool/abc-dragon.ydk", "decks/pool/adamancipator.ydk"]
    decks = [load_ydk(ROOT / n) for n in names]
    runs = {}
    for tag, a in (("x", GreedyPolicy(1)), ("y", GreedyPolicy(1)), ("z", RandomPolicy(5))):
        games = play_match(engine, decks, a, GreedyPolicy(2), [(0, 1)], range(2), num_envs=4, num_threads=2,
                           max_decisions=400)
        assert save_games(tmp_path / tag / "games.jsonl", games, names, "vs_bc") == 4
        runs[tag] = games
    rows = [json.loads(line) for line in (tmp_path / "x" / "games.jsonl").read_text().splitlines()]
    assert {(r["deck_a"], r["seed"], r["a_first"]) for r in rows} == {
        ("abc-dragon.ydk", s, f) for s in range(2) for f in (True, False)}
    assert all(r["decks"][r["a_seat"]] == names[0] for r in rows)  # seat-ordered deck paths
    m = _compare_mod()
    from ygo_zero.engine.duel import Engine

    eng = Engine()
    x, y, z = (m.load(tmp_path, t) for t in "xyz")
    assert set(x) == set(y) == set(z) and len(x) == 4
    for k in x:
        st = dict(x[k]["state"])
        f = m.fields(eng, st)
        assert len(st["actors"]) == len(st["choices"]) == len(st["turns"]) > 0
        assert f["mine"] or f["theirs"]
        assert m.divergence(st, dict(st)) is None
    # a different policy diverges somewhere (GreedyPolicy breaks ties randomly, so no x == y game check)
    diffs = [m.divergence(_filled(m, eng, x[k]),
                          _filled(m, eng, z[k])) for k in x]
    assert any(d is not None for d in diffs)


def _filled(m, eng, g):
    st = dict(g["state"])
    m.fields(eng, st)
    return st
