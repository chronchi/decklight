#!/usr/bin/env bash
# M0 speed profiling: latency + throughput + blocking-sync arms on one GPU box
# (docs/plans/2026-10-16-speed-and-cheap-tests.md §1.3/M0). Env: HOURS (1), MAX_DPH (0.50), MIN_CPUS (32),
# GA_SEEDS (6), LAT_MIN (8), B_MIN (8), C_MIN (6) (forwarded to the box), SMOKE=1 (short phases, 1 seed).
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl data/cache/value_p2/best.pt data/models/0546_26550M.tflite \
         data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=speed-profile-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} GA_SEEDS=${GA_SEEDS:-6} LAT_MIN=${LAT_MIN:-8} B_MIN=${B_MIN:-8} C_MIN=${C_MIN:-6} bash scripts/vast_speed_profile.sh $N" \
  --hours "${HOURS:-1}" --max-dph "${MAX_DPH:-0.50}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-20}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64}" \
  --data decks data/cards/emb data/models data/cache/P data/cache/value_p2 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
