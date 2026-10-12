#!/usr/bin/env bash
# Retrain stage 0f (docs/retrain/plan-v3.md; model v3 effect-key input): embed every card str1..16 + system string
# (21,039 keys, 11,511 unique texts) with Qwen3-Embedding-0.6B d512 -> runs/<name>/strings-qwen3-0.6b-d512.npz.
# Embeddings never run on the laptop (user 2026-10-11). ~11.5k short texts: minutes on any GPU; cheapest box.
cd "$(dirname "$0")/../.."
N=embed-strings-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N --cmd "MODE=strings bash scripts/vast_embed.sh $N" \
  --hours "${HOURS:-0.5}" --max-dph "${MAX_DPH:-0.15}" --min-cpus 4 --min-gpu-ram 8 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
