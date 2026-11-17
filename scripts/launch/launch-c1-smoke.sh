#!/usr/bin/env bash
# Cycle-1 smoke (docs/lab/prereg/2026-11-10-master-cycle1.md "Smoke"): two cycle-1 archetypes fine-tuned from X to it40,
# PAR 2, training only (STOP_AFTER_FT=1), on one GPU offer type; run once per type and compare $ per
# specialist-iteration. Usage: GPU=RTX_3080 MAX_DPH=0.13 MIN_CPUS=16 bash scripts/launch/launch-c1-smoke.sh
cd "$(dirname "$0")/../.."
X=runs/window-ppo-20261108-0940/train/export-X/policy
for f in $X/best.pt $X/metrics.jsonl data/cache/x_window/best.pt data/cache/value_p2/best.pt data/cache/P/best.pt \
         data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz data/cards/lua_desc.npz \
         data/models/0546_26550M.tflite decks/forge/c1/split.json; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
GPU=${GPU:?GPU=RTX_3080 or RTX_3090}
N=c1-smoke-$(echo "$GPU" | tr 'A-Z_' 'a-z-')-$(date +%Y%m%d-%H%M); echo $N
ENVS="STOP_AFTER_FT=1 REL=$X/best.pt SPLIT=decks/forge/c1/split.json ARCHS='${ARCHS:-primite-blue-eyes labrynth}'"
ENVS="$ENVS PACKS_IN=data/forge/none FT_ITERS=40 PAR=2 FT_HOURS=${FT_HOURS:-1.5}"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "$ENVS bash scripts/vast_forge_round.sh $N" \
  --hours "${HOURS:-2}" --max-dph "${MAX_DPH:?}" --min-cpus "${MIN_CPUS:-16}" --min-gpu-ram "${MIN_GPU_RAM:-10}" \
  --reserve "${RESERVE:-0.10}" --max-inet-cost "${MAX_INET:-0.005}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-46} gpu_name=$GPU" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/x_window data/cache/value_p2 $X \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
