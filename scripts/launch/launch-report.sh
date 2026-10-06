#!/usr/bin/env bash
# Report cards for data/cache/rc/* (zero = base, ft-teach, ft-teach-h, ft-search). Cap 1.5 h.
cd "$(dirname "$0")/../.."
N=rc-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/report.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N --cmd "bash scripts/vast_report.sh $N" \
  --hours 1.5 --max-dph 0.2 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=48" \
  --data data/cards/emb decks data/models data/demos data/cache/rc > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"; tail -3 data/cache/briefs/$N.launcher.log; cat runs/$N/summary.txt 2>/dev/null
