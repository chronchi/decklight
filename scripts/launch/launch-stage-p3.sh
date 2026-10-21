#!/usr/bin/env bash
# Stage P3: RL on the v3 network (user 2026-10-20: "I want to try rl on v3 ... to not having hand traps firing right at
# the beginning"; gate: "improved play at non inferiority at least"; pre-registered NOTEBOOK 2026-10-20 "Stage P3").
# Stage P's recipe (vast_stage_p.sh) from B-v3 (runs/stage-b-20261013-0617/v3/best.pt, copied to data/cache/bv3), KL
# anchor = B-v3; card reference REF = P on fresh deals (seeds 9000+). Verdict: scripts/audit/p3_gate.py on the laptop.
# Env: HOURS (8), MAX_DPH (0.40), SMOKE=1 (6 iterations + 1-seed card), ITERS (560), MS_IT (160), RL_HOURS (5.5), GA_SEEDS (24).
cd "$(dirname "$0")/../.."
for f in data/cache/bv3/best.pt data/cache/bv3/metrics.jsonl data/cache/P/best.pt data/cache/P/metrics.jsonl \
         data/models/0546_26550M.tflite data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz \
         data/cards/lua_desc.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=stage-p3-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} SMOKE_ITERS=${SMOKE_ITERS:-6} INIT=data/cache/bv3/best.pt REF=data/cache/P/best.pt DEAL0=9000 ITERS=${ITERS:-560} MS_IT=${MS_IT:-160} RL_HOURS=${RL_HOURS:-5.5} GA_SEEDS=${GA_SEEDS:-24} bash scripts/vast_stage_p.sh $N" \
  --hours "${HOURS:-8}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-24}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-48} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/bv3 data/cache/P > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
