#!/usr/bin/env bash
# Mid-game teacher data: DAgger (ygo-agent labels on our states) + window search. Generation only. Cap 2.5 h.
cd "$(dirname "$0")/../.."
N=t2-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/teachers.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N --cmd "bash scripts/vast_teachers.sh $N" \
  --hours 2.5 --max-dph 0.25 --min-cpus 32 --min-gpu-ram 16 --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data data/demos/teacher data/demos/gold data/cards/emb decks data/models data/cache/rl_init > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"; tail -3 data/cache/briefs/$N.launcher.log; cat runs/$N/teachers.txt 2>/dev/null
