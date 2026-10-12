#!/usr/bin/env bash
# Retrain stage M play-out anchor (docs/retrain/plan-v3.md §Stage M; plan cost $X): run after the stage M sweep
# (scripts/launch/run-stage-m.sh) finishes. Ships the turn records only (no OpenCode key: the LLM never runs on vast).
cd "$(dirname "$0")/../.."
[ "$(cat data/demos/turns/*/*.jsonl 2>/dev/null | wc -l)" -gt 0 ] || { echo "no turn records"; exit 3; }
N=turn-anchor-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N --cmd "bash scripts/vast_turn_anchor.sh $N" \
  --hours "${HOURS:-2}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data decks data/cards/emb data/cache/base_r3 data/cache/value_ppo3 data/demos/turns \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
