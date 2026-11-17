#!/usr/bin/env bash
# tsoj PPO steps 2+3 (NOTEBOOK 2026-11-07 tsoj PPO pre-registration): arms A (league + tsoj's bot) and B (league only)
# from forge_r1 on one GPU box, then the held-out tsoj gauntlet (scripts/vast_tsoj_ppo.sh). Env: ITERS (80), EXT_SHARE
# (0.2), CAP_T (5h, A), CAP_TB (2h, B), CAP_E (90m per arm), HOURS (9.5), MAX_DPH (0.40), MIN_CPUS (32), MIN_GPU_RAM (20), RESERVE (0.25), EXTRA_Q (more vast query terms, e.g. gpu_name=RTX_3090).
cd "$(dirname "$0")/../.."
for f in data/cache/forge_r1/best.pt data/cache/value_p2/best.pt data/cache/deck_opps/manifest.json \
         data/cards/emb/qwen3-0.6b-d512.npz; do [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
N=tsoj-ppo-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "ITERS=${ITERS:-80} EXT_SHARE=${EXT_SHARE:-0.2} CAP_T=${CAP_T:-5h} CAP_TB=${CAP_TB:-2h} CAP_E=${CAP_E:-90m} bash scripts/vast_tsoj_ppo.sh $N" \
  --hours "${HOURS:-9.5}" --reserve "${RESERVE:-0.25}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" \
  --min-gpu-ram "${MIN_GPU_RAM:-20}" --extra-query "cpu_ram>=64 ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 data/cache/deck_opps \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
