#!/usr/bin/env bash
# value_p2: more value-only RL from value_p (lr 1e-4) -> offline sanity -> P + hppo with value_p2 vs value_p, paired,
# vs ygo-agent on 1,488 fresh deals (seeds 9000+). User 2026-10-15: "Go with More value-only RL on value_p, with more
# iterations only for now". Env: HOURS (9), MAX_DPH (0.40), MIN_CPUS (32), VITERS (450), GA_SEEDS (6), VAR_CAP (210m).
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl data/cache/value_p/best.pt data/models/0546_26550M.tflite \
         data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
ls data/cache/pc_games/p/games-*.jsonl > /dev/null 2>&1 || { echo "no Pc games in data/cache/pc_games/p"; exit 3; }
N=value-p2-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} VITERS=${VITERS:-450} GA_SEEDS=${GA_SEEDS:-6} VAR_CAP=${VAR_CAP:-210m} bash scripts/vast_value_p2.sh $N" \
  --hours "${HOURS:-9}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-20}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64}" \
  --data decks data/cards/emb data/models data/cache/P data/cache/value_p data/cache/pc_games \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
