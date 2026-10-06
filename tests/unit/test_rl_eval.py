"""Scheduling tests for async RL eval (never loads real checkpoints or YgoAgentPolicy)."""

import json
import sys
import time
from pathlib import Path

from ygo_zero.rl.evaluate import AsyncEvaluator, format_eval_line


def _write_cmd(evals: Path, it: int, sleep: float = 0.0) -> list[str]:
    rec = {"iter": it, "eval": {"vs_bc": {"n": 2, "wr": 0.5, "ci": [0.1, 0.9], "cut": 0}},
           "eval_seconds": 1}  # fmt: skip
    line = json.dumps(rec)
    return [sys.executable, "-c",
            f"import json, time; time.sleep({sleep}); "
            f"open({str(evals)!r}, 'a').write({line!r} + chr(10))"]


def test_skip_while_running(tmp_path):
    ev = AsyncEvaluator(lambda it, ck: _write_cmd(tmp_path / "evals.jsonl", it, sleep=1.5), tmp_path)
    assert ev.maybe_start(1, "ckpt1") is True
    assert ev.maybe_start(2, "ckpt2") is False  # previous eval still running
    recs = ev.wait()
    assert [r["iter"] for r in recs] == [1]
    assert ev.maybe_start(2, "ckpt2") is True  # slot free again
    assert [r["iter"] for r in ev.wait()] == [2]


def test_record_relayed_once(tmp_path):
    ev = AsyncEvaluator(lambda it, ck: _write_cmd(tmp_path / "evals.jsonl", it), tmp_path)
    assert ev.maybe_start(5, "ckpt") is True
    recs = ev.wait()
    assert len(recs) == 1 and recs[0]["iter"] == 5
    assert ev.poll() == []  # relayed once: no duplicate
    assert ev.wait() == []
    line = format_eval_line(recs[0])
    assert line.startswith("[eval it 5]") and "vs_bc 0.500" in line


def test_failing_command_produces_alert(tmp_path):
    ev = AsyncEvaluator(lambda it, ck: [sys.executable, "-c", "import sys; sys.exit(1)"], tmp_path)
    assert ev.maybe_start(7, "ckpt") is True
    assert ev.wait() == []
    alert = (tmp_path / "ALERT").read_text()
    assert "ALERT train_rl: eval it 7 failed (exit 1), see eval-0007.txt" in alert
    assert (tmp_path / "eval-0007.txt").is_file()


def test_wait_returns_outstanding_and_poll_is_nonblocking(tmp_path):
    ev = AsyncEvaluator(lambda it, ck: _write_cmd(tmp_path / "evals.jsonl", it, sleep=1.0), tmp_path)
    assert ev.maybe_start(9, "ckpt") is True
    t = time.time()
    assert ev.poll() == []  # still running: nothing yet
    assert time.time() - t < 1.0  # poll did not block
    recs = ev.wait()  # blocks until the outstanding eval finishes
    assert [r["iter"] for r in recs] == [9]
