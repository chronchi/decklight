#!/usr/bin/env bash
# Retrain stage S (docs/retrain/plan-v2.md + plan-v3.md; user green light 2026-10-11 "yeah, go ahead"): honest-override
# search on base_r3 + value_ppo3 vs ygo-agent on the final card's 1,488 deals, paired with the stored hppo games; ho_t1
# on the 744 going-first deals; override replication audit. ~6,700 proc-min of search (2.27 proc-min per hppo game,
# x1.3 for adaptive samples) on ~37 procs -> ~3.5 h, ~$X at <= $X/h (48+ cores, 32 GB GPU as the final card).
cd "$(dirname "$0")/../.."
for f in data/cache/base_r3/best.pt data/cache/value_ppo3/best.pt data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
[ "$(ls data/cache/stage_s/hppo/games-*.jsonl | wc -l)" -gt 0 ] || { echo "missing data/cache/stage_s/hppo"; exit 3; }
N=stage-s-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N --cmd "bash scripts/vast_stage_s.sh $N" \
  --hours "${HOURS:-6}" --max-dph "${MAX_DPH:-0.38}" --min-cpus "${MIN_CPUS:-48}" --min-gpu-ram "${MIN_GPU_RAM:-32}" \
  --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data decks data/cards/emb data/cache/base_r3 data/cache/value_ppo3 data/models data/cache/stage_s \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
