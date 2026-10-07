#!/usr/bin/env bash
# Self-imitation step 2 data (user, 2026-10-07: "Agreed with the over fitting concerns, act accordingly. You can
# start"): turn-1 lines of base_r3 on ALL train decks (pool minus HOLDOUT minus kewl-tune), new hands (seed 1, the
# headroom run used seed 0), greedy + 16 samples each. Selection (scripts/select_turn1_lines.py --by field) and the
# read-through happen on the laptop before any training (user: check the games make sense at every step).
cd "$(dirname "$0")/../.."
N=selfimit-data-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt CRITIC=data/cache/critic_rl4/critic.pt SPLITS=train SEED=1 HANDS=16 K=16 MAX_TRAIN=0 T1_CAP=50m bash scripts/vast_turn1.sh $N" \
  --hours 1.5 --max-dph 0.3 --min-cpus 32 --min-gpu-ram 11 --extra-query "cpu_ram>=48 compute_cap>=800" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r3 data/cache/critic_rl4 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
