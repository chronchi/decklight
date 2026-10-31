#!/usr/bin/env bash
# Fresh Full-P (+ ho) arm on the product card's 1,488 gauntlet deals (seeds 9000+, GA 6, both seats). The Full-r1 card
# (runs/full-r1-20261030-1609) was INVALID by its drift check, so r1 is compared with today's Full-P instead of the
# stored 2026-10-21 arm (user 2026-10-31: "Fresh P arm, then decide"). NOTEBOOK 2026-10-31 "fresh Full-P arm".
# Budget: credit $X, floor $X, reserve $X -> HOURS 3.5 at <= $X/h.
cd "$(dirname "$0")/../.."
. scripts/product.sh
for f in "$PRODUCT_CKPT" "$PRODUCT_VALUE" data/cards/emb/qwen3-0.6b-d512.npz data/cards/lua_desc.npz data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=full-p-fresh-$(date +%Y%m%d-%H%M); echo $N
H="$PRODUCT_SEARCH --claim 1 --value $PRODUCT_VALUE"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=$PRODUCT_CKPT SM_ARGS='$H --deal-seed0 9000' VARIANTS='full_p|--opp ygoagent --ga-seeds 6' PROBE_NPROC=40 VAR_CAP=${VAR_CAP:-180m} GPU_MARGIN=${GPU_MARGIN:-1.1} bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-3.5}" --max-dph "${MAX_DPH:-0.28}" --min-cpus "${MIN_CPUS:-28}" --min-gpu-ram "${MIN_GPU_RAM:-20}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/value_p2 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
