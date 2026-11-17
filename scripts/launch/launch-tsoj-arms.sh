#!/usr/bin/env bash
# tsoj-arms (NOTEBOOK 2026-11-07): forced chain-0 activations vs P, opponent = tsoj's alpha-beta bot, on a cheap
# many-core box (scripts/vast_tsoj_arms.sh). Env: NODES (5000), SEEDS (22), CAP (60m per arm), HOURS (3.8),
# MAX_DPH (0.16), MIN_CPUS (32), RESERVE (0.25 credit kept), ARMS, NREP.
cd "$(dirname "$0")/../.."
for f in data/cache/forge_r1/best.pt data/cards/emb/qwen3-0.6b-d512.npz; do [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
N=tsoj-arms-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "NREP=${NREP:-40} ARMS='${ARMS:-top@1 rand@1}' NODES=${NODES:-5000} SEEDS=${SEEDS:-22} CAP=${CAP:-60m} bash scripts/vast_tsoj_arms.sh $N" \
  --hours "${HOURS:-3.8}" --reserve "${RESERVE:-0.25}" --max-dph "${MAX_DPH:-0.16}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 4 --min-compute-cap 0 \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
