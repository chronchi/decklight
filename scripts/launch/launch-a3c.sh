#!/usr/bin/env bash
# A3c played gate (docs/plans/2026-11-01-offline-app.md; NOTEBOOK 2026-11-02 "A3c"): Rust product vs the Python oracle,
# game for game. User 2026-11-02: "Use the 53 cents available here to make things, do we really need to top up?"
# -> no top-up: identity is exact, so Python replays only unit 0 of every pair; the box runs the plan
# (scripts/app/a3c_gate.py) in priority order until RUN_MIN, then the final check. CPU work on a GPU box (any GPU).
# Env: HOURS (2.25), RUN_MIN (105), MAX_DPH (0.14), RESERVE (0.10 = the credit floor), MIN_CPUS (32), SMOKE=1.
cd "$(dirname "$0")/../.."
for f in data/cache/onnx/r1/policy.onnx data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=a3c-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "RUN_MIN=${RUN_MIN:-105} SMOKE=${SMOKE:-0} bash scripts/app/vast_a3c.sh $N" \
  --hours "${HOURS:-2.25}" --max-dph "${MAX_DPH:-0.14}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 1 \
  --min-compute-cap 0 --reserve "${RESERVE:-0.10}" --extra-query "cpu_ram>=${MIN_RAM:-32}" \
  --data data/cache/onnx/r1 data/cards/emb \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
