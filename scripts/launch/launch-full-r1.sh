#!/usr/bin/env bash
# Full-r1 product card (user 2026-10-30: "Can you test r1 to become the product? That would be good"). Pre-registered in
# NOTEBOOK 2026-10-30 "Full-r1 product card"; the job is scripts/vast_full_r1.sh, the verdict scripts/audit/full_r1_gate.py.
# ~3,032 searched games (1,488 + 648 + 648 + 248). Budget: credit $X, floor $X -> HOURS 8 at <= $X/h.
cd "$(dirname "$0")/../.."
for f in data/cache/forge_r1/best.pt data/cache/forge_r1/metrics.jsonl data/cache/x4b_D/best.pt data/cache/P/best.pt \
         data/cache/value_p2/best.pt data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz \
         data/cards/lua_desc.npz data/models/0546_26550M.tflite data/forge/f3/pairs/trains.pairs; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=full-r1-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} bash scripts/vast_full_r1.sh $N" \
  --hours "${HOURS:-8}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-20}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/value_p2 data/cache/forge_r1 \
         data/cache/x4b_D data/forge/f3/pairs \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
