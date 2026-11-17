#!/usr/bin/env bash
# Step 1a deck qualification (docs/plans/2026-11-03-opp-turn-interaction.md), laptop, 4 single-core shards, ~1 h, $0.
set -euo pipefail
OUT=${1:-runs/combo-pool-$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
for i in 0 1 2 3; do
  OMP_NUM_THREADS=1 taskset -c $i uv run python scripts/audit/combo_pool.py --all --seeds 12 --samples 4 --hands 6 \
    --shard $i/4 --out "$OUT/s$i.json" > "$OUT/s$i.log" 2>&1 &
done
wait
grep -h ALERT "$OUT"/s*.log > "$OUT/ALERT" || true
echo "done $OUT"
