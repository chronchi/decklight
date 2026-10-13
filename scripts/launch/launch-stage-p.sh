#!/usr/bin/env bash
# Retrain stage P + Pc (plan-v1 §Stage P, plan-v2 §Stage P/Pc, plan-v3 change 2026-10-13: init and KL anchor = B-v1,
# user "B-v1 (Recommended)"): PPO with the ppo1 recipe, 560 iterations, it160 milestone, then the Pc card on seeds 3000+.
# Needs data/cache/bv1 (B-v1 best.pt + metrics.jsonl; written by launch-stage-b.sh from runs/stage-b1-20261013-0008/clean).
# Env: HOURS (7.5), MAX_DPH (0.40), SMOKE=1 (6 iterations + 1-seed card, ~25 min).
cd "$(dirname "$0")/../.."
for f in data/cache/bv1/best.pt data/cache/bv1/metrics.jsonl data/cache/base_r3/best.pt data/cache/base_r3/metrics.jsonl \
         data/models/0546_26550M.tflite data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=stage-p-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} ITERS=${ITERS:-560} RL_HOURS=${RL_HOURS:-5.5} GA_SEEDS=${GA_SEEDS:-24} bash scripts/vast_stage_p.sh $N" \
  --hours "${HOURS:-7.5}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-20}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-48}" \
  --data decks data/cards/emb data/models data/cache/bv1 data/cache/base_r3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
