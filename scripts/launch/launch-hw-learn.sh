#!/usr/bin/env bash
# Decisive-label learnability test (user 2026-10-20: "Yeah let's run the short first test"; pre-registration and gates
# in NOTEBOOK 2026-10-20 "hw learnability, pre-registered"). Fine-tunes P on the 1,483 Stage L decisive labels only and
# counts, on held-out labels, fixes vs breaks against P (scripts/eval_flips.py). No games. Credit floor $X, so a
# cheap box with a short cap.
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
[ "$(ls data/demos/sw-hw/*.jsonl | wc -l)" -eq 587 ] || { echo "sw-hw: expected 587 files"; exit 3; }
N=hwlearn-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "bash scripts/vast_hw_learn.sh $N" \
  --hours "${HOURS:-0.5}" --max-dph "${MAX_DPH:-0.15}" --min-cpus "${MIN_CPUS:-8}" --min-gpu-ram 8 \
  --extra-query "compute_cap>=800 cpu_ram>=24" \
  --data data/demos/sw-hw data/cards/emb decks data/cache/P > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
