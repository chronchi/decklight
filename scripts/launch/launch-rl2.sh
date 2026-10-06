#!/usr/bin/env bash
# RL round 2: PPO from base_r3 (KL re-anchored to it), --overlap, on an RTX 3090 (best samples/$ in bench2).
# 2.5 h of training, then the on-box report card vs base_r3 (top 2 checkpoints). Cap 3.5 h.
cd "$(dirname "$0")/../.."
N=rl2-$(date +%Y%m%d-%H%M); echo $N > data/cache/briefs/rl2.tag; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=data/cache/base_r3/best.pt RL_HOURS=2.5 EVAL_EVERY=40 RL_ARGS='--lr 4e-5 --overlap' bash scripts/vast_rl.sh $N" \
  --hours 3.5 --max-dph 0.25 --min-cpus 24 --min-gpu-ram 20 --extra-query "gpu_name=RTX_3090 cpu_ram>=48" \
  --data decks data/cards/emb data/models data/cache/base_r3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
