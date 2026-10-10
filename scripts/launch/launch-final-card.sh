#!/usr/bin/env bash
# Final report card, PLAN §0a P3–P4 (user 2026-10-10: "Let's go with 1 for now", after the ExIt pilot failed; decisions
# and gates in NOTEBOOK 2026-10-10 15:40). base_r3 + hppo vs ygo-agent on 1,488 fresh deals (paired with raw), guards on
# holdout (1,088 games) and new-card decks (1,080 games) vs raw base_r3, single-game latency.
# ~8,300 proc-min of search (P1: 2.27 proc-min per hppo game): a 40+ GB GPU with 48+ cores runs ~33 procs (P1 probe
# 772 MB GPU / 2 GB RSS per proc) -> ~4.5 h, ~$X at $X/h.
cd "$(dirname "$0")/../.."
for f in data/cache/base_r3/best.pt data/cache/value_ppo3/best.pt data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=final-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N --cmd "bash scripts/vast_final_card.sh $N" \
  --hours "${HOURS:-6.5}" --max-dph "${MAX_DPH:-0.45}" --min-cpus "${MIN_CPUS:-48}" --min-gpu-ram "${MIN_GPU_RAM:-40}" \
  --extra-query "compute_cap>=800 cpu_ram>=96" \
  --data decks data/cards/emb data/cache/base_r3 data/cache/value_ppo3 data/models > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
