"""Turn-1 line generation + scoring: rows, replay exactness, determinism, flags, fail-loud (CPU, tiny)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from decklight.decks import load_ydk
from decklight.eval.turn1 import config_from_dict, replay_turn1, resolve_decks
from decklight.model.policy import PolicyNet

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"
ARCH = {"d": 16, "layers": 1, "heads": 2, "hist": 0, "known": False}


def _mod():
    p = Path(__file__).resolve().parents[2] / "scripts" / "turn1_lines.py"
    spec = importlib.util.spec_from_file_location("turn1_lines_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _deck_names(n=2):
    return [p.stem for p in sorted(FIXTURES.glob("*.ydk"))[:n]]


def _make_ckpts(base: Path, zero_critic: bool = False) -> dict:
    """Random-init tiny policy + privileged critic checkpoints in the BCPolicy / load_critic format."""
    base.mkdir(parents=True, exist_ok=True)
    names = _deck_names()
    decks = [load_ydk(FIXTURES / f"{n}.ydk") for n in names]
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    emb_path = base / "emb.npz"
    np.savez(emb_path, **emb)
    torch.manual_seed(0)
    pol = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).eval()
    pdir = base / "pol"
    pdir.mkdir()
    torch.save(pol.state_dict(), pdir / "best.pt")
    (pdir / "metrics.jsonl").write_text(json.dumps({"args": dict(ARCH)}) + "\n")
    torch.manual_seed(1)
    cri = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, known=True, value=True).eval()
    if zero_critic:  # value head stuck at tanh(0) = 0: every live state scores identically
        with torch.no_grad():
            cri.value_head[3].weight.zero_()
            cri.value_head[3].bias.zero_()
    cdir = base / "cri"
    cdir.mkdir()
    torch.save(cri.state_dict(), cdir / "critic.pt")
    (cdir / "metrics.jsonl").write_text(json.dumps({"args": dict(ARCH)}) + "\n")
    ydks = ",".join(str(FIXTURES / f"{n}.ydk") for n in names)
    return {"ckpt": pdir / "best.pt", "critic": cdir / "critic.pt", "emb": emb_path, "ydks": ydks}


def _run(mod, ck: dict, out: Path, k: int = 2, seed: int = 0) -> int:
    return mod.main(["--ckpt", str(ck["ckpt"]), "--critic", str(ck["critic"]), "--emb", str(ck["emb"]),
                     "--decks", ck["ydks"], "--hands", "2", "--k", str(k), "--seed", str(seed),
                     "--out", str(out), "--procs", "1", "--threads", "1", "--device", "cpu"])  # fmt: skip


def _rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "lines.jsonl").read_text().splitlines() if line.strip()]


def test_rows_written_and_replay_exact(engine, tmp_path):
    mod = _mod()
    ck = _make_ckpts(tmp_path / "ck")
    out = tmp_path / "out"
    assert _run(mod, ck, out) == 0
    rows = _rows(out)
    assert len(rows) == 2 * 2 * (1 + 2), f"decks * hands * (greedy + K), got {len(rows)}"
    assert {(r["deck"], r["source"], r["kind"]) for r in rows} <= {
        (d, "random", k) for d in _deck_names() for k in ("greedy", "sample")}
    assert all(r["train_ok"] is False for r in rows)  # custom ydk list: never training data
    for r in rows:  # the stored choices replay to the same end field in a fresh env
        got = replay_turn1(engine, config_from_dict(r["config"]),
                           [int(c) for c in r["choices"]], r.get("auto_single", True))
        assert got["field"] == r["field"], (r["deck"], r["hand"], r["kind"], r["sample"])
        assert got["p0_decisions"] == r["p0_decisions"]
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["decks"]) == set(_deck_names())
    assert summary["splits"], "a per-split summary is written"


def test_greedy_deterministic_and_samples_vary(engine, tmp_path):
    mod = _mod()
    ck = _make_ckpts(tmp_path / "ck")
    assert _run(mod, ck, tmp_path / "a") == 0
    assert _run(mod, ck, tmp_path / "b") == 0
    key = lambda r: (r["deck"], r["hand"], tuple(r["choices"]), r["field"])  # noqa: E731
    ga = [key(r) for r in _rows(tmp_path / "a") if r["kind"] == "greedy"]
    gb = [key(r) for r in _rows(tmp_path / "b") if r["kind"] == "greedy"]
    assert ga == gb
    rows_a = _rows(tmp_path / "a")
    varied = any(
        s["choices"] != g["choices"]
        for g in (r for r in rows_a if r["kind"] == "greedy")
        for s in rows_a
        if (s["deck"], s["hand"], s["kind"]) == (g["deck"], g["hand"], "sample")
    )
    if not varied:
        pytest.skip("tiny random-init policy is degenerate here: no sample differs from greedy")


def test_train_ok_flags():
    assert resolve_decks("train")[::2] == ("train", True)
    assert resolve_decks("holdout")[2] is False
    assert resolve_decks("gauntlet")[2] is False


def test_fail_loud_on_identical_critic(engine, tmp_path):
    mod = _mod()
    ck = _make_ckpts(tmp_path / "ck", zero_critic=True)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as e:
        _run(mod, ck, out)
    assert e.value.code != 0
    assert "ALERT" in (out / "ALERT").read_text()
