#!/usr/bin/env bash
# Scale test (user, 2026-10-06: "if we just scale the iterations we see better results, we can spend even 1 dollar"):
# PPO from base_r3 with probe-B settings (no overlap, lr 6e-5, KL 0.05) for 4.5 h on a 3090, then the on-box card
# (top 2 by vs_bc vs base_r3). Cap 5.5 h, ≈ $1.
cd "$(dirname "$0")/../.."
N=rl3long-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=data/cache/base_r3/best.pt RL_HOURS=4.5 EVAL_EVERY=40 RL_ARGS='--lr 6e-5 --kl-coef 0.05' bash scripts/vast_rl.sh $N" \
  --hours 5.5 --max-dph 0.25 --min-cpus 24 --min-gpu-ram 20 --extra-query "gpu_name=RTX_3090 cpu_ram>=48" \
  --data decks data/demos data/cards/emb data/models data/cache/base_r3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
