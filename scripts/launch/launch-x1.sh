#!/usr/bin/env bash
# X1 (docs/plans/2026-10-27-experiments-and-rust.md; pre-registered NOTEBOOK 2026-10-27 "X1"): D0's recipe (PPO from
# P, value head from value_p, 160 iterations) plus advantage filtering (--adv-filter-q 0.75). The control is D0's
# stored P' (data/cache/D0 = runs/ppo-d0-20261015-1438/Pp), played raw on the same fresh deals (seeds 14000+), and the
# unseen-deck h2h is the X1 arm vs D0's P'. User 2026-10-27: "Yes" (start X1).
# Env: HOURS (3), MAX_DPH (0.40), MIN_CPUS (32), ITERS (160), Q (0.75), SMOKE=1 (4 iterations + 1-seed card).
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl data/cache/value_p/best.pt data/cache/D0/best.pt \
         data/models/0546_26550M.tflite data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=x1-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} ITERS=${ITERS:-160} RL_HOURS=${RL_HOURS:-2.5} GA_SEEDS=${GA_SEEDS:-24} VAR_CAP=${VAR_CAP:-60m} DEAL0=14000 CTRL=data/cache/D0/best.pt RL_EXTRA='--adv-filter-q ${Q:-0.75}' bash scripts/vast_ppo_d0.sh $N" \
  --hours "${HOURS:-3}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-48} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/models data/cache/P data/cache/value_p data/cache/D0 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
