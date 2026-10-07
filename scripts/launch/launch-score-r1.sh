#!/usr/bin/env bash
# Self-imitation round 2 line score (NOTEBOOK 2026-10-07 18:00: field count taught spending hand traps / quick
# effects): re-score the r1 hands (selfimit-data-20261007-1624, train decks only) by playing each candidate line on
# through the opponent's turn 2 (base_r3 both sides, 8 sampled rollouts, board@h2 = field + 0.5 hand + LP diff).
cd "$(dirname "$0")/../.."
N=score-r1-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt LINES=data/cache/selfimit_lines/r1.jsonl CANDS=4 SAMPLES=8 HORIZON=2 CAP=90m bash scripts/vast_score.sh $N" \
  --hours 2 --max-dph 0.3 --min-cpus 32 --min-gpu-ram 16 --extra-query "cpu_ram>=64 compute_cap>=800" \
  --data data/cards/emb decks data/models data/cache/base_r3 data/cache/selfimit_lines > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
