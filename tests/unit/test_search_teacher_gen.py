"""Search-teacher records (search_match --save-records, mixed --opp, --deal-mode train).

Two games per opponent kind; every record must load through dataset.py's loader as
window/window_own samples with soft targets, replay to the same game, and carry one
label per searched decision. Train deals never draw holdout decks.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pool_emb(tmp_path: Path) -> Path:
    """Random embeddings covering every pool + gauntlet card code (train deals draw real decks)."""
    from ygo_zero.decks import load_ydk

    files = sorted((ROOT / "decks" / "pool").glob("*.ydk")) + sorted((ROOT / "decks" / "gauntlet").glob("*.ydk"))
    codes = sorted({c for f in files for c in list(load_ydk(f).main) + list(load_ydk(f).extra)} - {0})
    assert codes
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    p = tmp_path / "emb.npz"
    np.savez(p, **emb)
    return p


def _tiny_ckpt(tmp_path: Path, emb: Path, tag: str, seed: int) -> Path:
    """A confident tiny ckpt (x8 weights): decisive play like a trained policy, so conf-skip
    fires selectively and laptop games stay fast."""
    from ygo_zero.model.policy import PolicyNet

    net = PolicyNet(dict(np.load(emb)), d=16, layers=1, heads=2, dropout=0.0).eval()
    torch.manual_seed(seed)
    for p in net.parameters():
        p.data.normal_().mul_(8.0)
    run = tmp_path / tag / "bc"
    run.mkdir(parents=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, run / "best.pt")
    (run / "metrics.jsonl").write_text(json.dumps({"args": {"d": 16, "layers": 1, "heads": 2}}) + "\n")
    return run / "best.pt"


def _run_kind(search_match, tmp_path: Path, emb: Path, ckpt: Path, kind: str, extra: list[str]) -> Path:
    """Two train-deal games vs `kind`; returns the records dir (under a `window/` dir for the mix kinds)."""
    out = tmp_path / f"m-{kind}"
    recdir = tmp_path / "window" / kind
    argv = ["search_match", "--ckpt", str(ckpt), "--emb", str(emb), "--opp", kind,
            "--deal-mode", "train", "--games", "2", "--seed-base", "100000",
            "--save-records", str(recdir),
            "--k", "2", "--samples", "1", "--horizon", "1",
            "--conf-skip", "0.9", "--own-frac", "1.0", "--own-turn1", "1",
            "--width", "2", "--batch", "16", "--out", str(out)] + extra  # fmt: skip
    old = sys.argv
    sys.argv = argv
    try:
        search_match.main()
    finally:
        sys.argv = old
    return recdir


def _check_records(engine, recdir: Path, decks_dir: Path | list[Path]) -> None:
    """Records replay from their own config, load as window/window_own soft-target samples."""
    from ygo_zero.demos.dataset import iter_samples, record_config, sample_kind, search_soft_targets
    from ygo_zero.decks import load_ydk
    from ygo_zero.engine.cpp_env import CppEnv

    recs = [json.loads(x) for f in sorted(recdir.glob("*.jsonl"))
            for x in f.read_text().splitlines() if x.strip()]
    assert len(recs) == 2
    for rec in recs:  # deck stems name the config's seat decks (catches seat-swap bugs)
        for seat in (0, 1):
            dirs = decks_dir if isinstance(decks_dir, list) else [decks_dir]
            d = load_ydk(next(x / f"{rec['decks'][seat]}.ydk" for x in dirs if (x / f"{rec['decks'][seat]}.ydk").is_file()))
            assert list(d.main) == rec["config"][f"deck{seat}_main"]
            assert list(d.extra) == rec["config"][f"deck{seat}_extra"]
    for rec in recs:
        n = len(rec["choices"])
        assert len(rec["actors"]) == len(rec["turns"]) == n
        assert len(rec["targets"]) == len(rec["labels"]) == n
        # replay from the record's own config reproduces the game
        env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
        obs = env.reset()
        for c in rec["choices"]:
            assert obs is not None
            obs = env.step(int(c))
        assert (obs is None) == (not rec["truncated"])
        if not rec["truncated"]:
            assert int(env.winner) == rec["winner"]
        # one label per searched decision; the target is the move actually played
        searched = [e["k"] for e in rec["search"]]
        labelled = [k for k, lb in enumerate(rec["labels"]) if lb != "skip"]
        assert sorted(searched) == sorted(labelled) and labelled
        for e in rec["search"]:
            assert rec["labels"][e["k"]] == ("window" if e["kind"] == "window" else "own")
            assert rec["targets"][e["k"]] == rec["choices"][e["k"]] == e["chosen"]
            assert len(e["cands"]) == len(e["scores"]) >= 2
        # dataset.py sees soft targets at exactly the searched decisions
        soft = search_soft_targets(rec)
        assert sorted(soft) == sorted(searched)
        for k in searched:
            cands, scores = soft[k]
            assert len(cands) == len(scores) >= 2
    samples = list(iter_samples(engine, recdir))
    assert samples
    kinds = {m["kind"] for _, _, _, m in samples}
    assert kinds <= {"window", "window_own"} and kinds
    n_soft = sum(1 for _, _, _, m in samples if "soft" in m)
    assert n_soft == sum(1 for rec in recs for lb in rec["labels"] if lb != "skip")
    for _, _, _, m in samples:
        assert (m["kind"] == "window") == (m["label"] == "window")
        assert sample_kind(recdir / "x.jsonl", m["label"]) == m["kind"]


def _check_attribution(tmp_path: Path, kind: str, recdir: Path) -> None:
    """Teacher records carry the game's seat-ordered decks; each file holds its S deck's games."""
    rows = [json.loads(x) for x in (tmp_path / f"m-{kind}" / "games-0.jsonl").read_text().splitlines()
            if x.strip()]
    by_idx = {r["game_idx"]: r for r in rows}
    n = 0
    for f in sorted(recdir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            row = by_idx[int(rec["game_id"].split("-")[1])]
            assert rec["decks"] == row["decks"]
            assert f.stem == rec["decks"][row["s_seat"]]
            n += 1
    assert n == 2


def _holdout() -> set[str]:
    return set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}


def test_teacher_gen_self_ckpt_greedy(engine, tmp_path):
    search_match = _load("search_match")
    emb = _pool_emb(tmp_path)
    ckpt = _tiny_ckpt(tmp_path, emb, "base", 0)
    opp_ckpt = _tiny_ckpt(tmp_path, emb, "opp", 1)
    hold = _holdout()
    for kind, extra in (("self", []), ("ckpt", ["--opp-ckpt", str(opp_ckpt)]), ("greedy", [])):
        recdir = _run_kind(search_match, tmp_path, emb, ckpt, kind, extra)
        _check_records(engine, recdir, ROOT / "decks" / "pool")
        _check_attribution(tmp_path, kind, recdir)
        # train deals never draw holdout decks (match rows carry the seat decks)
        rows = [json.loads(x) for x in (tmp_path / f"m-{kind}" / "games-0.jsonl").read_text().splitlines()
                if x.strip()]
        assert len(rows) == 2
        for r in rows:
            assert not (set(r["decks"]) & hold), r["decks"]
            if kind != "self":
                assert r["opp_decisions"] > 0
                assert "ya_decisions" not in r


def test_teacher_gen_ygoagent(engine, tmp_path):
    model = ROOT / "data/models/0546_26550M.tflite"
    codes = ROOT / "data/models/code_list.txt"
    if not model.is_file() or not codes.is_file():
        pytest.skip("ygo-agent weights not on this machine")
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        pytest.skip("ai_edge_litert not installed")
    search_match = _load("search_match")
    emb = _pool_emb(tmp_path)
    ckpt = _tiny_ckpt(tmp_path, emb, "base", 0)
    recdir = _run_kind(search_match, tmp_path, emb, ckpt, "ygoagent", [])
    _check_records(engine, recdir, [ROOT / "decks" / "pool", ROOT / "decks" / "gauntlet"])
    _check_attribution(tmp_path, "ygoagent", recdir)
    rows = [json.loads(x) for x in (tmp_path / "m-ygoagent" / "games-0.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    # user, 2026-10-08: our side pilots pool train decks; only ygo-agent plays gauntlet lists
    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    gauntlet = {p.stem for p in (ROOT / "decks/gauntlet").glob("*.ydk")}
    for r in rows:
        assert (ROOT / "decks/pool" / f"{r['deck_s']}.ydk").is_file() and r["deck_s"] not in hold
        assert r["deck_r"] in gauntlet
    assert all(r["ya_decisions"] > 0 for r in rows)


def test_train_schedule_never_draws_holdout():
    search_match = _load("search_match")
    hold = {"kewl-tune"}
    names = ["a", "b", "c"]
    sched = search_match.train_schedule(names, 50, 100000, hold)
    assert len(sched) == 50
    for di, dj, s, s_seat, idx in sched:
        assert names[di] not in hold and names[dj] not in hold
        assert di != dj and s == 100000 + idx and s_seat == idx % 2
    with pytest.raises(SystemExit):  # every pair holds a holdout stem: the first draw fails
        search_match.train_schedule(["kewl-tune", "x"], 4, 7, {"kewl-tune", "x"})


def test_teacher_gen_claim_queue(engine, tmp_path):
    """--save-records turns on the dynamic claim queue: two procs on one out dir never play the same game,
    together play every game, and a proc that finds every game claimed exits cleanly with 0 games."""
    search_match = _load("search_match")
    emb = _pool_emb(tmp_path)
    ckpt = _tiny_ckpt(tmp_path, emb, "base", 0)
    out = tmp_path / "m-self"
    for proc in (0, 1):  # sequential: proc 0 claims all 3 games, proc 1 finds none left
        _run_kind(search_match, tmp_path, emb, ckpt, "self",
                  ["--games", "3", "--nproc", "2", "--proc", str(proc)])
    rows = [json.loads(x) for p in sorted(out.glob("games-*.jsonl")) for x in p.read_text().splitlines() if x.strip()]
    idx = [r["game_idx"] for r in rows]
    assert sorted(idx) == [0, 1, 2], idx
    assert sorted(int(p.name) for p in (out / "claims").iterdir()) == [0, 1, 2]


def test_s_seat_only_and_turn_limits(engine, tmp_path):
    """--s-seat-only 1 keeps the full schedule's game idx (P2 crash 2026-10-09: sched[idx] IndexError); with
    --max-search-turn/--stop-turn every game stops after turn 2 with S in seat 1."""
    search_match = _load("search_match")
    emb = _pool_emb(tmp_path)
    ckpt = _tiny_ckpt(tmp_path, emb, "base", 0)
    _run_kind(search_match, tmp_path, emb, ckpt, "greedy",
              ["--s-seat-only", "1", "--max-search-turn", "2", "--stop-turn", "2", "--games", "4"])
    rows = [json.loads(x) for x in (tmp_path / "m-greedy" / "games-0.jsonl").read_text().splitlines() if x.strip()]
    assert len(rows) == 2 and all(r["s_seat"] == 1 for r in rows)
    assert all(r["turns"] <= 2 for r in rows)
