#!/usr/bin/env bash
# Report card of rl3long checkpoints (iter 640 / 360, the vs_bc peaks) vs base_r3; re-run because rl3long's own card
# stage had no data/demos (NOTEBOOK 2026-10-07 00:25). Same playtest flags as card-ppo1. Cap 1.5 h.
cd "$(dirname "$0")/../.."
N=card-rl3long-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r3/best.pt CKPTS='data/cache/rl3long_card/i0640 data/cache/rl3long_card/i0360' PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards' bash scripts/vast_card.sh $N" \
  --hours 1.5 --max-dph 0.25 --min-cpus 24 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r3 data/cache/rl3long_card > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
