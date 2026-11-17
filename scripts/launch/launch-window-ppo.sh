#!/usr/bin/env bash
# Window PPO (NOTEBOOK 2026-11-08 window PPO pre-registration): arm X from A, low KL, no KL on opponent-turn windows,
# forced activations in heavy combo-line episodes (scripts/vast_window_ppo.sh). Stages data/cache/window_ppo from
# tsoj-ppo-20261107-2123 (export-A, eval-A/P) and window-gate-20261104-0735 (stored window rows) when missing.
# Env: ITERS (80), KL (0.03), DAGS_P (0.5), EXT_SHARE (0.2), RL_HOURS (3.3), CAP_T (4h), CAP_E (70m), HOURS (5.25), MAX_DPH (0.40),
# MIN_CPUS (32), MIN_GPU_RAM (20), RESERVE (0.25), EXTRA_Q (more vast query terms).
cd "$(dirname "$0")/../.."
W=data/cache/window_ppo; R=runs/tsoj-ppo-20261107-2123
if [ ! -s $W/A/policy/best.pt ]; then
  mkdir -p $W/wg && cp -r $R/train/export-A $W/A && cp $R/eval-A.jsonl $R/eval-P.jsonl $W/ \
    && cp runs/window-gate-20261104-0735/wg/s[0-9]*.json $W/wg/ || { echo "staging $W failed"; exit 3; }
fi
python3 scripts/build_heavy_lines.py data/cache/line_starts/starts-v2.jsonl --out data/cache/line_starts/starts-v2-heavy.jsonl >/dev/null || exit 3
for f in data/cache/forge_r1/best.pt data/cache/value_p2/best.pt data/cards/emb/qwen3-0.6b-d512.npz $W/A/policy/best.pt \
         $W/A/value/best.pt $W/eval-A.jsonl $W/eval-P.jsonl data/cache/line_starts/starts-v2-heavy.jsonl; do
  [ -s "$f" ] || { echo "missing $f"; exit 3; }
done
[ "$(ls $W/wg/s[0-9]*.json | wc -l)" = 30 ] || { echo "expected 30 window shards in $W/wg"; exit 3; }
N=window-ppo-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "ITERS=${ITERS:-80} KL=${KL:-0.03} DAGS_P=${DAGS_P:-0.5} EXT_SHARE=${EXT_SHARE:-0.2} RL_HOURS=${RL_HOURS:-3.3} CAP_T=${CAP_T:-4h} CAP_E=${CAP_E:-70m} bash scripts/vast_window_ppo.sh $N" \
  --hours "${HOURS:-5.25}" --reserve "${RESERVE:-0.25}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" \
  --min-gpu-ram "${MIN_GPU_RAM:-20}" --extra-query "cpu_ram>=64 gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 \
         data/cache/line_starts data/cache/window_ppo \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
