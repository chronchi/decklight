#!/usr/bin/env bash
# Kolibri §5 checkpoint merging on rl5 (user 2026-10-08: "Agreed with your points, go ahead with this a-c"):
# mix50 = 0.5 base_r3 + 0.5 it215, soupRL = uniform average of it40/80/120/160/200/215 (scripts/merge_ckpts.py).
# Both play the same schedules as rl5ga/rl5vb (31 gauntlet decks, ring 4, both seats, seeds 0-2 = 744 games) vs
# ygo-agent (runs/<N>/ga) and vs base_r3 (runs/<N>/vb), every game saved; base games come from rl5ga/rl5vb it0
# (iter-0000 == base_r3 bit for bit), paired by deal with scripts/audit/rl_ga_compare.py.
cd "$(dirname "$0")/../.."
M=data/cache/rl5_merge
for f in $M/mix50/best.pt $M/soupRL/best.pt data/cache/base_r3/best.pt; do [ -s $f ] || { echo "missing $f"; exit 3; }; done
N=rl5merge-$(date +%Y%m%d-%H%M); echo $N
SPECS="mix50=$M/mix50/best.pt soupRL=$M/soupRL/best.pt"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SEEDS=3 bash scripts/vast_ga_compare.sh $N/ga '$SPECS'; SEEDS=3 VS_BASE=data/cache/base_r3/best.pt bash scripts/vast_ga_compare.sh $N/vb '$SPECS'" \
  --hours 1.0 --max-dph 0.25 --min-cpus 32 --min-gpu-ram 10 --extra-query "cpu_ram>=48" \
  --data decks data/cards/emb data/models data/cache/base_r3 $M > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
