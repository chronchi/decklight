#!/usr/bin/env bash
# Encoding diagnostics before the retrain plan (user 2026-10-11: run the cheap diagnostics before the planner; "make
# sure that embeddings in card text capture the timing of the cards"): policy/value input-flip tests and linear probes
# on base_r3 / value_ppo3 / a random-init control, plus card-text timing probes vs bag-of-words.
cd "$(dirname "$0")/../.."
for f in data/cache/base_r3/best.pt data/cache/value_ppo3/best.pt data/cache/diag/text_labels.npz; do [ -f "$f" ] || { echo "missing $f"; exit 3; }; done
N=diag-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "POLICY=data/cache/base_r3/best.pt VALUE=data/cache/value_ppo3/best.pt bash scripts/vast_diag.sh $N" \
  --hours 1.5 --max-dph 0.3 --min-cpus 16 --min-gpu-ram 8 --extra-query "cpu_ram>=32 compute_cap>=800" \
  --data data/cache/diag data/cards/emb data/cache/base_r3 data/cache/value_ppo3 decks > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
