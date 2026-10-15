#!/usr/bin/env bash
# Step A / D0 (scripts/vast_ppo_d0.sh): continue PPO from P (value head from value_p), 160 iterations + card on deal
# seeds 8000+. User 2026-10-15: "Yes" (start with step A). Env: HOURS (3), MAX_DPH (0.40), MIN_CPUS (32), ITERS, SMOKE.
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl data/cache/value_p/best.pt \
         data/models/0546_26550M.tflite data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=ppo-d0-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} ITERS=${ITERS:-160} RL_HOURS=${RL_HOURS:-2.5} GA_SEEDS=${GA_SEEDS:-24} VAR_CAP=${VAR_CAP:-60m} bash scripts/vast_ppo_d0.sh $N" \
  --hours "${HOURS:-3}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-48}" \
  --data decks data/cards/emb data/models data/cache/P data/cache/value_p \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
