#!/usr/bin/env bash
# M2 turn-1 search teacher (ADR 0006): combo check + 3000 searched hands + distill (ft-search) + report cards. Cap 3 h.
cd "$(dirname "$0")/../.."
N=s1-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/search.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N --cmd "bash scripts/vast_search.sh $N" \
  --hours 3 --max-dph 0.25 --min-cpus 16 --min-gpu-ram 16 --extra-query "compute_cap>=800 cpu_ram>=48" \
  --data data/demos data/cards/emb decks data/models data/cache/rl_init > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"; tail -4 data/cache/briefs/$N.launcher.log; cat runs/$N/search.txt runs/$N/timing.txt runs/$N/summary.txt 2>/dev/null
for f in runs/$N/playtest-*.txt; do echo "== $f"; grep -E "^combo mean|^vs " "$f"; done
