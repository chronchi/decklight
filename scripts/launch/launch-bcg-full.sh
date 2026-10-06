#!/usr/bin/env bash
# Full gold BC run (after batch 4 + muse-eval-4). Cap 4.5 h, <= $X/h; box destroyed at the end.
cd "$(dirname "$0")/../.."
N=bcg-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N --cmd "bash scripts/vast_bc_gold.sh $N" \
  --hours 4.5 --max-dph 0.25 --min-cpus 8 --min-gpu-ram 8 --extra-query "compute_cap>=800" \
  --data data/demos data/cards/emb decks data/cache/bc_base > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"; tail -4 data/cache/briefs/$N.launcher.log; cat runs/$N/timing.txt runs/$N/summary.txt 2>/dev/null
for f in runs/$N/playtest-*.txt; do echo "== $f"; grep -E "^combo|^vs " "$f"; done
