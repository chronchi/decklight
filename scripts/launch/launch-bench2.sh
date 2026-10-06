#!/usr/bin/env bash
# Speed plan P3 + P6: update profile, fused/compile configs and a 5-iteration train_rl (base_r3) on one GPU type.
# Usage: launch-bench2.sh <RTX_3090|RTX_4090|RTX_5090> <max dph> [RL_ARGS]. Cap 45 min.
cd "$(dirname "$0")/../.."
G=$1; D=$2; A=${3:-}
N=bench2-$(echo $G | tr 'A-Z_' 'a-z-')-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r3/best.pt SECTIONS='profile update' BENCH_S=20 RL_ITERS=5 RL_ARGS='$A' bash scripts/vast_bench_rl.sh $N" \
  --hours 0.75 --max-dph $D --min-cpus 16 --min-gpu-ram 20 --extra-query "gpu_name=$G cpu_ram>=48" \
  --data decks data/cards/emb data/cache/base_r3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
