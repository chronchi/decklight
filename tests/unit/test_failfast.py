"""Fail-fast invariants (AGENTS.md "Fail fast and loud"): checkpoint selection, identical-weights guard, ALERT relay."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_select_on_last_prefers_any_trained_epoch_over_init():
    train_bc = _load("train_bc")
    init = {"epoch": -1, "val": {"acc": 0.9, "per_kind": {"gold": 0.9}}}
    ep0 = {"epoch": 0, "val": {"acc": 0.1, "per_kind": {"gold": 0.1}}}
    assert train_bc.select_score(ep0, "last", -1.0) > train_bc.select_score(init, "last", -1.0)
    assert train_bc.select_score({"merged_last": 3}, "last", -1.0) == -1.0  # a merge never displaces the last epoch
    assert train_bc.select_score(ep0, "val:gold", -1.0) < train_bc.select_score(init, "val:gold", -1.0)


def test_same_weights_guard(tmp_path):
    playtest = _load("playtest")
    a = {"w": torch.ones(3, dtype=torch.float16)}
    torch.save(a, tmp_path / "a.pt")
    torch.save({"model": {"w": torch.ones(3)}, "opt": {}}, tmp_path / "b.pt")
    torch.save({"w": torch.zeros(3, dtype=torch.float16)}, tmp_path / "c.pt")
    assert playtest.same_weights(tmp_path / "a.pt", tmp_path / "b.pt")
    assert not playtest.same_weights(tmp_path / "a.pt", tmp_path / "c.pt")


def test_relay_alerts_appends_only_new_lines(tmp_path, monkeypatch):
    from ygo_zero.infra import vast

    remote = ["ALERT train-ft-dw: best.pt is the --init weights"]

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout="\n".join(remote) + "\n", stderr="")

    monkeypatch.setattr(vast.subprocess, "run", fake_run)
    log = tmp_path / "launcher.log"
    seen = vast.relay_alerts("h", 1, "/r", log, 0)
    remote.append("playtest-ft-dw skipped")
    seen = vast.relay_alerts("h", 1, "/r", log, seen)
    seen = vast.relay_alerts("h", 1, "/r", log, seen)
    lines = log.read_text().splitlines()
    assert seen == 2 and lines == ["ALERT train-ft-dw: best.pt is the --init weights", "ALERT playtest-ft-dw skipped"]
