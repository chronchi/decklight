#!/usr/bin/env bash
# P's value net (value-only RL on P's frozen trunk) -> offline leaf choice vs value_ppo3 -> P raw and P + hppo search vs
# ygo-agent on 1,488 fresh deals (seeds 5000+). User 2026-10-13: "P's critic, value-only RL (Recommended)".
# Env: HOURS (5), MAX_DPH (0.40), MIN_CPUS (32), VITERS (150), SMOKE=1 (~35 min).
cd "$(dirname "$0")/../.."
PRUN=${PRUN:-runs/stage-p-20261013-1240}
for f in "$PRUN/P/best.pt" "$PRUN/P/metrics.jsonl" data/cache/value_ppo3/best.pt data/models/0546_26550M.tflite \
         data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
ls "$PRUN"/p/games-*.jsonl > /dev/null 2>&1 || { echo "no Pc games in $PRUN/p"; exit 3; }
mkdir -p data/cache/P data/cache/pc_games/p
cp "$PRUN/P/best.pt" "$PRUN/P/metrics.jsonl" data/cache/P/
cp "$PRUN"/p/games-*.jsonl data/cache/pc_games/p/
N=value-p-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} VITERS=${VITERS:-150} GA_SEEDS=${GA_SEEDS:-6} VAR_CAP=${VAR_CAP:-170m} bash scripts/vast_value_p.sh $N" \
  --hours "${HOURS:-5}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-20}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64}" \
  --data decks data/cards/emb data/models data/cache/P data/cache/value_ppo3 data/cache/pc_games \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
