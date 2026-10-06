#!/usr/bin/env bash
# Game audit of base_r3 (ppo1 iter 561): the full report card vs ft-swv with every game saved + rendered on the box.
cd "$(dirname "$0")/../.."
N=audit-r3-$(date +%Y%m%d-%H%M); echo $N > data/cache/briefs/audit-r3.tag; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r2/best.pt CKPTS='data/cache/audit_r3/r3' RENDER=1 PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards --save-games 1000' bash scripts/vast_card.sh $N" \
  --hours 1 --max-dph 0.20 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=32" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r2 data/cache/audit_r3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
