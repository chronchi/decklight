"""Hint bank: 2 train-split starts with a tiny policy (K=1) banks rows or fails loud."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load(name: str):
    p = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_mod", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_ckpt(tmp_path: Path):
    """A real (random) d=16 BC checkpoint + emb over two train pool decks' codes."""
    from ygo_zero.decks import load_ydk
    from ygo_zero.model.policy import PolicyNet

    decks = [load_ydk(ROOT / "decks/pool/abc-dragon.ydk"),
             load_ydk(ROOT / "decks/pool/adamancipator.ydk")]
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    run = tmp_path / "run" / "bc"
    run.mkdir(parents=True)
    np.savez(tmp_path / "emb.npz", **emb)
    net = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).eval()
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, run / "best.pt")
    (run / "metrics.jsonl").write_text(json.dumps({"args": {"d": 16, "layers": 1, "heads": 2}}) + "\n")
    return run / "best.pt", tmp_path / "emb.npz"


def _starts_file(tmp_path: Path) -> Path:
    """2 train-split starts (empty prefix: full games from the deal)."""
    from ygo_zero.decks import load_ydk

    rows = []
    for name in ("abc-dragon", "adamancipator"):
        d = load_ydk(ROOT / "decks/pool" / f"{name}.ydk")
        e = load_ydk(ROOT / "decks/pool/agents.ydk")
        rows.append({"main0": list(d.main), "extra0": list(d.extra),
                     "main1": list(e.main), "extra1": list(e.extra),
                     "seed": 7, "hand": 5, "draw": 1, "lp": 8000, "setup": [],
                     "prefix": [], "src": "test", "turn": 1,
                     "game_id": f"test-{name}", "decks": [name, "agents"]})
    p = tmp_path / "starts.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _run_bank(tmp_path: Path, **kw):
    hint_bank = _load("hint_bank")
    ckpt, emb = _tiny_ckpt(tmp_path)
    out = tmp_path / "bank"
    argv = ["--ckpt", str(ckpt), "--emb", str(emb), "--starts", str(_starts_file(tmp_path)),
            "--n", "2", "--games", "1", "--seed", "0", "--k", "2", "--samples", "1",
            "--horizon", "1", "--width", "2", "--batch", "16", "--max-decisions", "60",
            "--out", str(out),
            *[x for kv in kw.items() for x in (f"--{kv[0]}", str(kv[1]))]]
    try:
        hint_bank.main(argv)
    except SystemExit as e:
        return hint_bank, out, e.code
    return hint_bank, out, 0


def test_hint_bank_banks_or_fails_loud(engine, tmp_path):
    """2 starts, K=1: either rows are banked (replay yields legal actions) or the run
    exits non-zero with the expected message (0 hard starts, or hard starts with no
    hinted win)."""
    hint_bank, out, code = _run_bank(tmp_path)
    if code != 0:
        assert code == 1
        assert (out / "games-0.jsonl").is_file()
        recs = [json.loads(x) for x in (out / "games-0.jsonl").read_text().splitlines()
                if x.strip()]
        kinds = {r["kind"] for r in recs}
        assert "raw" in kinds  # either 0 hard starts, or hard starts with no hinted win
        return  # the expected fail-loud path (no bank.jsonl written)
    bank = out / "bank.jsonl"
    assert bank.is_file()
    rows = [json.loads(x) for x in bank.read_text().splitlines() if x.strip()]
    assert rows, "exit 0 with 0 banked rows"
    # Replay of banked rows yields legal actions (the train_rl auxiliary-loss read path).
    from ygo_zero.engine.duel import Engine
    from ygo_zero.rl.hints import load_bank

    eng = Engine()
    got = load_bank(bank, eng, hist=0, known=False)
    assert got["n_decisions"] > 0 and len(got["act"]) == got["n_decisions"]
    assert got["start_ids"] and got["rows"] == len(rows)
    summ = json.loads((out / "summary.json").read_text())
    assert summ["decisions_banked"] == got["n_decisions"]
    assert summ["hard"] >= 1 and summ["games_banked"] >= 1


def test_summarize_fails_loud_on_no_hard_or_no_bank(tmp_path):
    hint_bank = _load("hint_bank")
    out = tmp_path / "o"
    out.mkdir()
    (out / "games-0.jsonl").write_text(
        json.dumps({"start_id": 0, "kind": "raw", "actor_win": True}) + "\n")
    with pytest.raises(SystemExit) as e:
        hint_bank.summarize(out)
    assert e.value.code == 1  # 0 hard starts
    # Hard starts but no hinted win: bank files hold no won row.
    (out / "games-0.jsonl").write_text(
        "\n".join(json.dumps({"start_id": 0, "kind": k, "actor_win": False})
                  for k in ("raw", "hinted")) + "\n")
    (out / "bank-0.jsonl").write_text("")
    with pytest.raises(SystemExit):
        hint_bank.summarize(out)  # 0 banked decisions


def test_summarize_applies_gates(tmp_path):
    """--min-wins and the second-opponent gate: only starts passing both are banked (both opponents' wins)."""
    hint_bank = _load("hint_bank")
    out = tmp_path / "g"
    out.mkdir()
    (out / "gates.json").write_text(json.dumps({"min_wins": 2, "opp2_min_wins": 1, "n_opps": 2}))
    games, bank = [], []

    def add(start, actor, wins0, wins1):
        games.extend({"start_id": start, "actor": actor, "kind": "raw", "actor_win": False} for _ in range(4))
        for opp, w in ((0, wins0), (1, wins1)):
            for g in range(4):
                games.append({"start_id": start, "actor": actor, "kind": "hinted", "opp": opp,
                              "actor_win": g < w})
                if g < w:
                    bank.append({"start_id": start, "actor_seat": actor, "opp": opp, "actor_win": True,
                                 "config": {}, "prefix": [0], "choices": [0, 1, 2], "actors": [actor] * 3})

    add(0, 0, 2, 1)  # passes
    add(0, 1, 2, 0)  # same start, other seat: fails the opp2 gate
    add(1, 0, 1, 4)  # one lucky win vs opp 0: fails min_wins
    (out / "games-0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in games))
    (out / "bank-0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bank))
    summ = hint_bank.summarize(out)
    rows = [json.loads(x) for x in (out / "bank.jsonl").read_text().splitlines() if x.strip()]
    assert {(r["start_id"], r["actor_seat"]) for r in rows} == {(0, 0)}
    assert len(rows) == 3 and summ["pass_all_gates"] == 1 and summ["pass_min_wins"] == 2
    assert summ["hard"] == 3 and summ["hard_with_hinted_win"] == 3
    assert summ["decisions_banked"] == 3 * 2  # the prefix decision is not counted
    (out / "gates.json").write_text(json.dumps({"min_wins": 4, "opp2_min_wins": 1, "n_opps": 2}))
    with pytest.raises(SystemExit):
        hint_bank.summarize(out)  # nothing reaches 4/4


def test_slot_candidates_gates(tmp_path):
    hint_bank = _load("hint_bank")
    starts = _starts_file(tmp_path)
    ids = ["test-abc-dragon", "test-adamancipator"]

    def slot(s, seat, cls, n, w, lp, me, opp, game=None):
        return {"slot": 2 * s + seat, "start": s, "seat": seat, "cls": cls, "n": n, "w": w, "lp_me": lp,
                "cnt": {"me": me, "opp": opp}, "game": game or ids[s]}

    even = {"hand": 3, "mzone": 1}
    rows = [slot(0, 0, "hopeless", 8, 0, 8000, even, even),             # in
            slot(0, 1, "trainable", 8, 1, 8000, even, even),            # in (p 0.125)
            slot(1, 0, "trainable", 8, 4, 8000, even, even),            # out: p 0.5
            slot(1, 1, "hopeless", 8, 0, 1000, even, even),             # out: LP
            slot(1, 1, "hopeless", 8, 0, 8000, {"hand": 1}, {"hand": 2, "mzone": 2, "szone": 1}),  # out: 4 behind
            slot(1, 0, "solved", 8, 8, 8000, even, even)]               # out: solved
    p = tmp_path / "slots.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = hint_bank.slot_candidates(p, starts, 10, 0, 0.25, 1001, 3)
    assert sorted((s, seat) for s, _, seat in got) == [(0, 0), (0, 1)]
    done = tmp_path / "done"
    done.mkdir()
    (done / "games-0.jsonl").write_text(json.dumps({"start_id": 0, "actor": 1, "kind": "raw", "actor_win": False}) + "\n")
    got = hint_bank.slot_candidates(p, starts, 10, 0, 0.25, 1001, 3, exclude=done)
    assert [(s, seat) for s, _, seat in got] == [(0, 0)]
    p.write_text(json.dumps(slot(0, 0, "hopeless", 8, 0, 8000, even, even, game="other")) + "\n")
    with pytest.raises(SystemExit):
        hint_bank.slot_candidates(p, starts, 10, 0, 0.25, 1001, 3)  # slots from another starts file


def test_replay_rows_skips_prefix_decisions(engine, tmp_path):
    """Bank rows carry the DAGS prefix; its decisions are the source game's and are never distilled."""
    from ygo_zero.eval.turn1 import config_to_dict, dags_config, replay_prefix
    from ygo_zero.rl.hints import replay_rows

    starts = [json.loads(x) for x in _starts_file(tmp_path).read_text().splitlines()]

    cfg = dags_config(starts[0])
    choices, actors = [], []
    env, obs = replay_prefix(engine, cfg, [])
    for _ in range(6):
        actors.append(int(env.player))
        choices.append(0)
        obs = env.step(0)
        assert obs is not None
    actor = actors[0]
    row = {"config": config_to_dict(cfg), "choices": choices, "actors": actors, "actor_seat": actor}
    _, full = replay_rows(engine, [row], hist=0, known=False)
    _, cut = replay_rows(engine, [dict(row, prefix=choices[:3])], hist=0, known=False)
    assert len(full) == sum(a == actor for a in actors)
    assert len(cut) == sum(a == actor for a in actors[3:])
