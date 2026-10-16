#!/usr/bin/env bash
# Aux evaluator heads N3: P-vs-P games + aux shards + train_aux on the frozen value_p2 trunk.
# Small GPU box (RTX 3060-class or similar, >= 16 cores, max $X/h, 2 h cap).
# Env: HOURS (2), MAX_DPH (0.20), MIN_CPUS (24), MIN_GPU_RAM (8), MIN_RAM (32),
# GAMES (12000), SEED (5000000), THREADS (8), NUM_ENVS (256), PROBE_GAMES (256), GAMES_CAP (50m),
# STRIDE (4), EPOCHS (10), P/VAL overrides.
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/value_p2/best.pt data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=aux-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "P=${P:-data/cache/P/best.pt} VAL=${VAL:-data/cache/value_p2/best.pt} GAMES=${GAMES:-12000} SEED=${SEED:-5000000} THREADS=${THREADS:-8} NUM_ENVS=${NUM_ENVS:-256} PROBE_GAMES=${PROBE_GAMES:-256} GAMES_CAP=${GAMES_CAP:-50m} STRIDE=${STRIDE:-4} EPOCHS=${EPOCHS:-10} bash scripts/vast_aux.sh $N" \
  --hours "${HOURS:-2}" --max-dph "${MAX_DPH:-0.20}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-8}" \
  --extra-query "compute_cap>=${COMPUTE_CAP:-750} cpu_ram>=${MIN_RAM:-32}" \
  --data decks data/cards/emb data/models data/cache/P data/cache/value_p2 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
