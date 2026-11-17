#!/usr/bin/env bash
# Small product card for X (user 2026-11-08: "Let's go with option 1, it might be enough"): the product search
# (scripts/product.sh: honest k4 s4 h1, det-tries 32, skip, conf-skip 0.9, ho) with policy X and each value net,
# vs ygo-agent on the first 744 of the 1,488 fresh deals (deal seeds 9000-9002, ga-seeds 3, both seats):
#   x_v2: X + value_p2 (the current product value)   x_vx: X + X's window-PPO critic (data/cache/card_x/Xv)
# Paired on the laptop afterwards: x_vx - x_v2 (value_x needed?), x_v2 - stored r1 product (runs/full-r1-20261030-1609
# /full_r1, same deals) = the policy switch. Pre-registered rule: NOTEBOOK 2026-11-08 21:30. ~$X at <= $X/h.
cd "$(dirname "$0")/../.."
. scripts/product.sh
for f in data/cache/card_x/X/ckpt/best.pt data/cache/card_x/Xv/best.pt "$PRODUCT_VALUE" data/models/0546_26550M.tflite \
         data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
cmp -s data/cache/card_x/Xv/best.pt "$PRODUCT_VALUE" && { echo "X critic == value_p2"; exit 3; }
N=product-x-$(date +%Y%m%d-%H%M); echo $N
GS=${GA_SEEDS:-3}; X=data/cache/card_x/X/ckpt/best.pt
V="x_v2|--opp ygoagent --ga-seeds $GS --deal-seed0 9000 --value $PRODUCT_VALUE;x_vx|--opp ygoagent --ga-seeds $GS --deal-seed0 9000 --value data/cache/card_x/Xv/best.pt"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=$X SM_ARGS='$PRODUCT_SEARCH --claim 1' VARIANTS='$V' PROBE_NPROC=40 VAR_CAP=${VAR_CAP:-80m} GPU_MARGIN=1.3 bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-2.6}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-20}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX" \
  --data decks data/cards/emb data/models data/cache/card_x data/cache/value_p2 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
