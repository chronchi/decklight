"""Re-run a running train_rl job's async evals that died with CUDA OOM, on the same box.

Why: on 24 GB GPUs (A10, 3090) the async eval subprocess can OOM while the update holds ~18 GB
(rl4-20261007-0956 it 80, NOTEBOOK 2026-10-07 12:20). A failed eval is never recorded, so the card stage
(which picks checkpoints by their evals in metrics.jsonl) only sees the evals that survived.

Loop until train.txt says "exit": for each rl/eval-NNNN.txt that contains "out of memory" and whose
iteration has no record yet, re-run scripts/eval_ckpt.py with the job's own eval args (read from the
metrics.jsonl args record), retrying on failure. Success → the record is appended to rl/metrics.jsonl
(train_rl appends there too, one flushed line at a time) and announced in runs/<name>/ALERT
("ALERT retry_evals: [eval it N] ...") so the launcher relays it.

Usage (on the box, from /workspace/ygo-zero):
  nohup uv run --no-sync python scripts/retry_evals.py runs/<name> > runs/<name>/retry_evals.txt 2>&1 &
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.rl.evaluate import format_eval_line  # noqa: E402


def main() -> int:
    run = Path(sys.argv[1])
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rl = run / "rl"
    first = json.loads((rl / "metrics.jsonl").read_text().splitlines()[0])
    a, names = first["args"]["rl"], first["eval_decks"]
    tries: dict[int, int] = {}
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    while True:
        done_training = "\nexit " in (run / "train.txt").read_text()
        have = set()
        for ln in (rl / "metrics.jsonl").read_text().splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if "eval" in r and "iter" in r:
                have.add(r["iter"])
        todo = []
        for f in sorted(rl.glob("eval-*.txt")):
            it = int(re.findall(r"\d+", f.stem)[0])
            if (it not in have and tries.get(it, 0) < 10 and "out of memory" in f.read_text()
                    and (rl / f"iter-{it:04d}.pt").is_file()):
                todo.append(it)
        for it in todo:
            tries[it] = tries.get(it, 0) + 1
            out = rl / "evals-retry.jsonl"
            cmd = [sys.executable, str(ROOT / "scripts" / "eval_ckpt.py"), "--ckpt", str(rl / f"iter-{it:04d}.pt"),
                   "--init", a["init"], "--iter", str(it), "--emb", a["emb"], "--eval-names", ",".join(names),
                   "--gauntlet-decks", a["gauntlet_decks"], "--eval-seeds", a["eval_seeds"],
                   "--ga-seeds", a["ga_seeds"], "--ga-opps", a["ga_opps"], "--device", a["device"],
                   "--threads", str(threads), "--out", str(out)]  # fmt: skip
            if a.get("no_ygoagent") == "True":
                cmd.append("--no-ygoagent")
            n0 = len(out.read_text().splitlines()) if out.is_file() else 0
            rc = subprocess.run(cmd, env=env, stdout=open(rl / f"eval-{it:04d}.retry.txt", "w"),  # noqa: SIM115
                                stderr=subprocess.STDOUT, check=False).returncode  # fmt: skip
            new = out.read_text().splitlines()[n0:] if out.is_file() else []
            print(f"it {it} try {tries[it]} rc {rc} records {len(new)}", flush=True)
            if rc != 0 or not new:
                if tries[it] >= 10:
                    with open(run / "ALERT", "a") as f:
                        f.write(f"ALERT retry_evals: eval it {it} still failing after 10 tries\n")
                continue
            rec = json.loads(new[-1])
            with open(rl / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            with open(run / "ALERT", "a") as f:
                f.write(f"ALERT retry_evals: {format_eval_line(rec)}\n")
        if done_training and not todo:
            return 0
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
