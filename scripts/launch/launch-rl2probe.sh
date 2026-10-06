#!/usr/bin/env bash
# rl2 diagnosis (NOTEBOOK 2026-10-06 18:45): short PPO probes from base_r3 on a 3090, no report card, 0.75 h.
# Usage: launch-rl2probe.sh <tag> "<train_rl args>". Compare eval vs_bc at it 40/80/120 with rl2 (0.500/0.512/0.515).
cd "$(dirname "$0")/../.."
T=$1; A=$2
N=rl2p-$T-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=data/cache/base_r3/best.pt RL_HOURS=0.75 EVAL_EVERY=40 CARD=0 RL_ARGS='$A' bash scripts/vast_rl.sh $N" \
  --hours 1.4 --max-dph 0.25 --min-cpus 24 --min-gpu-ram 20 --extra-query "gpu_name=RTX_3090 cpu_ram>=48" \
  --data decks data/cards/emb data/models data/cache/base_r3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
