#!/usr/bin/env bash
# ADR 0007 round 1 training: one capped box per variant, in parallel (ft-t13 control, ft-swv, ft-swd), from the base
# in data/cache/base_r1; report card per variant (vs base, greedy, random, ygo-agent, new-card decks, combo).
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M); : > data/cache/briefs/r1.tags
for V in ${VARIANTS:-ft-t13 ft-swv ft-swd}; do
  N=r1-$V-$TS; echo $N | tee -a data/cache/briefs/r1.tags
  taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
    --cmd "BASE=data/cache/base_r1/best.pt VARIANTS='$V' PLAY_ALL=1 SKIP_SCRATCH=1 PT_ZERO=0 PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards' bash scripts/vast_bc_gold.sh $N" \
    --hours 3.5 --max-dph 0.15 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=64" \
    --data data/demos data/cards/emb decks data/models data/cache/base_r1 > data/cache/briefs/$N.launcher.log 2>&1 &
  sleep 20  # stagger offer picks so the jobs do not race for the same offer
done
wait
for N in $(cat data/cache/briefs/r1.tags); do
  echo "== $N"; tail -2 data/cache/briefs/$N.launcher.log; cat runs/$N/ALERT 2>/dev/null
  for f in runs/$N/playtest-*.txt; do grep -E "^combo mean|^vs " "$f"; done
done
