#!/usr/bin/env bash
# Value v2 step 0 (docs/lab/prereg/2026-11-11-value-v2.md): Y (observable-history product policy) + value_p2 in the product
# search vs ygo-agent, 96 games (deal seeds 9000+, both seats). Pass = games finish, no ALERT, searched decisions > 0,
# determinization accepts worlds (obs_history is part of the root-obs check now). Raw Y on the same deals for scale.
cd "$(dirname "$0")/../.."
. scripts/product.sh
for f in "$PRODUCT_CKPT" "$PRODUCT_CKPT_DIR/metrics.jsonl" "$PRODUCT_VALUE" data/models/0546_26550M.tflite data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=y-search-smoke-$(date +%Y%m%d-%H%M); echo $N
L="--opp ygoagent --ga-seeds 1 --deal-seed0 9000 --games-limit ${GAMES:-96}"
V="y_v2|$L --value $PRODUCT_VALUE;y_raw|$L --raw 1"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=$PRODUCT_CKPT SM_ARGS='$PRODUCT_SEARCH --claim 1' VARIANTS='$V' PROBE_NPROC=40 VAR_CAP=${VAR_CAP:-25m} GPU_MARGIN=1.3 bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-1.0}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --max-inet-cost "${MAX_INET:-0.015}" --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-60} gpu_name!=CMP_170HX" \
  --data decks data/cards/emb data/models $PRODUCT_CKPT_DIR $PRODUCT_VALUE_DIR \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
