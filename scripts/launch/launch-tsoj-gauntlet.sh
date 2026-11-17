#!/usr/bin/env bash
# tsoj gauntlet step 1 (NOTEBOOK 2026-11-07 gauntlet pre-registration): node sweep + mirror gauntlet, on a cheap
# many-core box (scripts/vast_tsoj_gauntlet.sh). Env: SWEEP, SWEEP_SEEDS, MAIN_SEEDS, CAP_SWEEP, CAP_MAIN, HOURS (3.5),
# MAX_DPH (0.16), MIN_CPUS (32), RESERVE (0.25 credit kept).
cd "$(dirname "$0")/../.."
for f in data/cache/forge_r1/best.pt data/cards/emb/qwen3-0.6b-d512.npz; do [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
N=tsoj-gauntlet-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SWEEP='${SWEEP:-500 1000 2000 5000}' SWEEP_SEEDS=${SWEEP_SEEDS:-4} MAIN_SEEDS=${MAIN_SEEDS:-28} CAP_SWEEP=${CAP_SWEEP:-40m} CAP_MAIN=${CAP_MAIN:-100m} bash scripts/vast_tsoj_gauntlet.sh $N" \
  --hours "${HOURS:-3.5}" --reserve "${RESERVE:-0.25}" --max-dph "${MAX_DPH:-0.16}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 4 --min-compute-cap 0 \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
