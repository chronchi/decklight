#!/usr/bin/env bash
# Report card of the base ft-swv (base_r2) under the loop guard: the reference for round 2's variants.
cd "$(dirname "$0")/../.."
N=r2-zero-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/zero2.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r2/best.pt VARIANTS='zero' SKIP_SCRATCH=1 PT_ZERO=1 PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards' bash scripts/vast_bc_gold.sh $N" \
  --hours 2 --max-dph 0.15 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r2 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
