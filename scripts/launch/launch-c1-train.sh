#!/usr/bin/env bash
# Cycle-1 training (docs/lab/prereg/2026-11-10-master-cycle1.md, amendments 2-3): fine-tune + pack + validate a share of
# the c1 archetypes from X (STOP_AFTER_PACK=1), the meta lists in the opponent pool (EXTRA_POOL=1). Run once per box,
# each with its own ARCHS; the packs are merged on the laptop for the verify / distill / card job.
# Usage: GPU=RTX_A5000 MAX_DPH=0.20 PAR=2 ARCHS='a b ...' bash scripts/launch/launch-c1-train.sh
cd "$(dirname "$0")/../.."
X=runs/window-ppo-20261108-0940/train/export-X/policy
for f in $X/best.pt $X/metrics.jsonl data/cache/x_window/best.pt data/cache/value_p2/best.pt data/cache/P/best.pt \
         data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz data/cards/lua_desc.npz \
         data/models/0546_26550M.tflite decks/forge/c1/split.json; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
GPU=${GPU:?GPU=RTX_A5000 or RTX_3090}; ARCHS=${ARCHS:?ARCHS='arch ...'}
N=c1-train-$(echo "$GPU" | tr 'A-Z_' 'a-z-')-$(date +%Y%m%d-%H%M); echo $N
ENVS="STOP_AFTER_PACK=1 EXTRA_POOL=1 REL=$X/best.pt REL_NAME=X SPLIT=decks/forge/c1/split.json ARCHS='$ARCHS'"
ENVS="$ENVS PACKS_IN=data/forge/none FT_ITERS=40 PAR=${PAR:-2}"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "$ENVS bash scripts/vast_forge_round.sh $N" \
  --hours "${HOURS:-5.5}" --max-dph "${MAX_DPH:?}" --min-cpus "${MIN_CPUS:-14}" --min-gpu-ram "${MIN_GPU_RAM:-22}" \
  --reserve "${RESERVE:-0.10}" --max-inet-cost "${MAX_INET:-0.005}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-46} gpu_name=$GPU" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/x_window data/cache/value_p2 $X \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
