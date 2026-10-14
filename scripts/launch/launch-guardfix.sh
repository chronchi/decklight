#!/usr/bin/env bash
# Re-measure P raw and P + hppo + value_p after the LoopGuard fix (scripts/vast_guardfix.sh). User 2026-10-14:
# "Yes, fix and rerun". Env: HOURS (4), MAX_DPH (0.40), MIN_CPUS (32), GA_SEEDS (4), VAR_CAP (130m).
cd "$(dirname "$0")/../.."
VPR=${VPR:-runs/value-p-20261014-0726/value_p}
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl "$VPR/best.pt" "$VPR/metrics.jsonl" \
         data/models/0546_26550M.tflite data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
mkdir -p data/cache/value_p && cp "$VPR/best.pt" "$VPR/metrics.jsonl" data/cache/value_p/
N=guardfix-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "GA_SEEDS=${GA_SEEDS:-4} VAR_CAP=${VAR_CAP:-130m} bash scripts/vast_guardfix.sh $N" \
  --hours "${HOURS:-4}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-20}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64}" \
  --data decks data/cards/emb data/models data/cache/P data/cache/value_p \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
