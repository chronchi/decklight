#!/usr/bin/env bash
# Per-deck fine-tune pilot: fine-tune P + value_p2 on each deck in DECKS, then a raw + searched
# per-deck card vs ygo-agent (scripts/vast_ft_pilot.sh on the box).
# Pre-registered: NOTEBOOK 2026-10-22 per-deck fine-tune pilot.
# Env: HOURS (4, 0.75 when SMOKE=1), MAX_DPH (0.32), SMOKE=1 (3 iters + 1-seed cards),
#   DECKS, FT_HOURS, SHARE, RAW_SEEDS, SEARCH_SEEDS, SMOKE_ITERS, NP, GB_PER_PROC, RAM_PER_PROC, MS_IT, MS_MIN
#   (passed through to vast_ft_pilot.sh; see its header for defaults).
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl data/cache/value_p2/best.pt data/cache/value_p2/metrics.jsonl \
         data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz \
         data/cards/lua_desc.npz data/models/0546_26550M.tflite decks/gauntlet/BlueEyes.ydk decks/gauntlet/Branded.ydk; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=ft-pilot-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
HH=${HOURS:-4}; [ "${SMOKE:-0}" = 1 ] && HH=${HOURS:-0.75}
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} SMOKE_ITERS=${SMOKE_ITERS:-3} DECKS='${DECKS:-BlueEyes Branded}' FT_HOURS=${FT_HOURS:-0.75} SHARE=${SHARE:-0.5} RAW_SEEDS=${RAW_SEEDS:-8} SEARCH_SEEDS=${SEARCH_SEEDS:-4} NP='${NP:-}' GB_PER_PROC=${GB_PER_PROC:-1.5} RAM_PER_PROC=${RAM_PER_PROC:-3} MS_IT=${MS_IT:-20} MS_MIN=${MS_MIN:-0.45} bash scripts/vast_ft_pilot.sh $N" \
  --hours "$HH" --max-dph "${MAX_DPH:-0.32}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-22}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/value_p2 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
