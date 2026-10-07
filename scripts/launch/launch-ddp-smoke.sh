#!/usr/bin/env bash
# Multi-GPU smoke (user, 2026-10-07: "Then also have muse implement multi GPU training"): rl4's exact config on a
# 2-GPU box (NGPU=2, torchrun) for 0.25 h of RL, no card. Checks DDP runs on real GPUs (checksums, evals, league
# sync) and measures s/iteration vs rl4's ~72 s on one A10. Cap 1 h at <= $X/h.
cd "$(dirname "$0")/../.."
[ -f data/cache/value_ppo3/best.pt ] || { echo "missing data/cache/value_ppo3"; exit 3; }
N=ddpsmoke-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "NGPU=2 INIT=data/cache/base_r3/best.pt RL_HOURS=0.25 EVAL_EVERY=10 CARD=0 RL_ARGS='--lr 6e-5 --kl-coef 0.05 --league 1 --p-self 0.4 --p-ygoagent 0.15 --snap-every 10 --league-size 6 --priv-critic 1 --critic-init data/cache/value_ppo3/best.pt' bash scripts/vast_rl.sh $N" \
  --hours 1 --num-gpus 2 --max-dph 0.5 --min-cpus 40 --min-gpu-ram 20 --extra-query "cpu_ram>=64" \
  --data decks data/demos data/cards/emb data/models data/cache/base_r3 data/cache/value_ppo3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
