"""Smoke test for scripts/bench_rl.py (CPU, tiny model, ~1 s per measurement)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_bench_rl_smoke(tmp_path):
    out = tmp_path / "b.jsonl"
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts/bench_rl.py"), "--smoke", "--device", "cpu",
         "--sections", "env", "infer", "rollout", "update", "--seconds", "1", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=240, check=False,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    got = {r["section"] for r in recs}
    assert {"env", "infer", "rollout", "update"} <= got, sorted(got)
    assert all(r["rate"] is not None and r["rate"] > 0 for r in recs if "skipped" not in r)
