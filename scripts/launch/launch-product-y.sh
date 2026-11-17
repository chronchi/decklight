#!/usr/bin/env bash
# Small product card for Y (user 2026-11-11: "The 50 cents card"): Y + value_p2 in the product search (scripts/product.sh,
# unchanged since launch-product-x.sh) vs ygo-agent on the same 744 deals as X's card (deal seeds 9000-9002, ga-seeds 3,
# both seats). Paired on the laptop: y_v2 - stored x_v2 (runs/product-x-20261108-2059/x_v2), scripts/audit/paired_sm.py.
# Pre-registered rule: NOTEBOOK 2026-11-11 (Y product card). ~$X at <= $X/h.
cd "$(dirname "$0")/../.."
. scripts/product.sh
for f in "$PRODUCT_CKPT" "$PRODUCT_CKPT_DIR/metrics.jsonl" "$PRODUCT_VALUE" data/models/0546_26550M.tflite \
         data/cards/emb/qwen3-0.6b-d512.npz runs/product-x-20261108-2059/x_v2; do
  [ -e "$f" ] || { echo "missing $f"; exit 3; }
done
N=product-y-$(date +%Y%m%d-%H%M); echo $N
V="y_v2|--opp ygoagent --ga-seeds ${GA_SEEDS:-3} --deal-seed0 9000 --value $PRODUCT_VALUE"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=$PRODUCT_CKPT SM_ARGS='$PRODUCT_SEARCH --claim 1' VARIANTS='$V' PROBE_NPROC=40 VAR_CAP=${VAR_CAP:-80m} GPU_MARGIN=1.3 bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-1.8}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --max-inet-cost "${MAX_INET:-0.015}" --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-60} gpu_name!=CMP_170HX" \
  --data decks data/cards/emb data/models $PRODUCT_CKPT_DIR $PRODUCT_VALUE_DIR \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
