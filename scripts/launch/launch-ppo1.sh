#!/usr/bin/env bash
# PPO from ft-swv (base_r2) with the fast loop (async env, bf16, mb1024, cached ref). 2.5 h of training, cap 3.3 h.
cd "$(dirname "$0")/../.."
N=ppo1-$(date +%Y%m%d-%H%M); echo $N > data/cache/briefs/ppo1.tag; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=data/cache/base_r2/best.pt RL_HOURS=2.5 EVAL_EVERY=40 RL_ARGS='--lr 4e-5' bash scripts/vast_rl.sh $N" \
  --hours 3.3 --max-dph 0.40 --min-cpus 16 --min-gpu-ram 20 --extra-query "total_flops>=70 cpu_ram>=48" \
  --data decks data/cards/emb data/models data/cache/base_r2 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
