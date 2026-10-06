#!/usr/bin/env bash
# Report card of ppo1 checkpoints (iter 320 / 400 / 561) vs ft-swv, same playtest flags as r2-zero. Cap 1.5 h.
cd "$(dirname "$0")/../.."
N=card-ppo1-$(date +%Y%m%d-%H%M); echo $N > data/cache/briefs/card-ppo1.tag; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r2/best.pt CKPTS='data/cache/ppo1_card/i0320 data/cache/ppo1_card/i0400 data/cache/ppo1_card/i0561' PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards' bash scripts/vast_card.sh $N" \
  --hours 1.5 --max-dph 0.25 --min-cpus 24 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r2 data/cache/ppo1_card > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
