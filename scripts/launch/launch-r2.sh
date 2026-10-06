#!/usr/bin/env bash
# ADR 0007 round 2 training: one capped box per variant (ft-r2 teacher 0.5, ft-r2t teacher 0.25) from ft-swv
# (data/cache/base_r2); report card per variant, all under the loop guard (reference: r2-zero card of the base).
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M); : > data/cache/briefs/r2.tags
for V in ${VARIANTS:-ft-r2 ft-r2t}; do
  N=r2-$V-$TS; echo $N | tee -a data/cache/briefs/r2.tags
  taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
    --cmd "BASE=data/cache/base_r2/best.pt VARIANTS='$V' PLAY_ALL=1 SKIP_SCRATCH=1 PT_ZERO=0 PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards' bash scripts/vast_bc_gold.sh $N" \
    --hours 3.5 --max-dph 0.15 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=64" \
    --data data/demos data/cards/emb decks data/models data/cache/base_r2 > data/cache/briefs/$N.launcher.log 2>&1 &
  sleep 20
done
wait
