#!/usr/bin/env bash
# Showcase games, product vs product (scripts/vast_showcase.sh). User 2026-10-15: "generate a few games ... so I can
# share with someone". Small box: 3 single-game processes. Env: HOURS (1.5), MAX_DPH (0.25), PAIRS.
cd "$(dirname "$0")/../.."
. scripts/product.sh
for f in "$PRODUCT_CKPT" data/cache/P/metrics.jsonl "$PRODUCT_VALUE" data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=showcase-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "PAIRS=${PAIRS:-mitsurugi-meta1:snake-eye,branded:tearlaments,blue-eyes:sky-striker} bash scripts/vast_showcase.sh $N" \
  --hours "${HOURS:-1.5}" --max-dph "${MAX_DPH:-0.25}" --min-cpus "${MIN_CPUS:-8}" --min-gpu-ram "${MIN_GPU_RAM:-8}" \
  --extra-query "compute_cap>=750 cpu_ram>=${MIN_RAM:-16}" \
  --data decks data/cards/emb data/models data/cache/P "$PRODUCT_VALUE_DIR" \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
