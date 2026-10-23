#!/usr/bin/env bash
# Fine-tune learning curve: fine-tune BlueEyes for 2 h with snapshots, validation card on
# every snapshot, best-on-validation pick, test-card confirmation (scripts/vast_ft_curve.sh).
# Pre-registered: NOTEBOOK 2026-10-22 fine-tune learning curve.
# Env: HOURS (2.75, 0.75 when SMOKE=1), MAX_DPH (0.19), SMOKE=1 (4 iters + 1-seed cards),
#   DECK, FT_HOURS, SNAP_EVERY, SHARE, VAL_SEED0, VAL_SEEDS, TEST_SEED0, TEST_SEEDS,
#   NP, GB_PER_PROC, RAM_PER_PROC, MS_IT, MS_MIN (passed through to vast_ft_curve.sh;
#   see its header for defaults).
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl data/cache/value_p2/best.pt data/cache/value_p2/metrics.jsonl \
          data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz \
          data/cards/lua_desc.npz data/models/0546_26550M.tflite decks/gauntlet/BlueEyes.ydk; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=ft-curve-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
HH=${HOURS:-2.75}; [ "${SMOKE:-0}" = 1 ] && HH=${HOURS:-0.75}
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} DECK='${DECK:-BlueEyes}' FT_HOURS=${FT_HOURS:-2} SNAP_EVERY=${SNAP_EVERY:-40} SHARE=${SHARE:-0.5} VAL_SEED0=${VAL_SEED0:-7000} VAL_SEEDS=${VAL_SEEDS:-6} TEST_SEED0=${TEST_SEED0:-9000} TEST_SEEDS=${TEST_SEEDS:-8} NP='${NP:-}' GB_PER_PROC=${GB_PER_PROC:-0.6} RAM_PER_PROC=${RAM_PER_PROC:-3} MS_IT=${MS_IT:-20} MS_MIN=${MS_MIN:-0.45} bash scripts/vast_ft_curve.sh $N" \
  --hours "$HH" --max-dph "${MAX_DPH:-0.19}" --min-cpus "${MIN_CPUS:-24}" \
  --min-gpu-ram "${MIN_GPU_RAM:-15}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/value_p2 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
