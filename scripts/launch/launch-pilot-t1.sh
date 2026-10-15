#!/usr/bin/env bash
# Plan-v4 pilot tranche 1 (scripts/vast_pilot_t1.sh): P-distilled BC student + raw card. User 2026-10-15: "Let's go with
# the test, just added some credits". Env: HOURS (4), MAX_DPH (0.45), MIN_CPUS (32), GAMES (20000), EPOCHS (4), SMOKE.
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl data/cache/bv1/best.pt data/cache/base_r3/best.pt \
         data/models/0546_26550M.tflite data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=pilot-t1-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} GAMES=${GAMES:-20000} EPOCHS=${EPOCHS:-4} GA_SEEDS=${GA_SEEDS:-24} VAR_CAP=${VAR_CAP:-60m} bash scripts/vast_pilot_t1.sh $N" \
  --hours "${HOURS:-4}" --max-dph "${MAX_DPH:-0.45}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-24}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-90}" \
  --data decks data/cards/emb data/models data/cache/P data/cache/bv1 data/cache/base_r3 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
