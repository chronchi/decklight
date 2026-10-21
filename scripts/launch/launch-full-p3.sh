#!/usr/bin/env bash
# Full product with the v3 policy (user 2026-10-21: "can we now try v3 as the full product along with search and value
# head? We need to do the comparisons now again so we use a better model right in terms of plays"). Pre-registered in
# NOTEBOOK 2026-10-21 "Full-P3 card".
#   full_p3: P3 (data/cache/P3 = runs/stage-p3-20261021-0048/P) + value_p2 + the product search, vs ygo-agent, on the
#            SAME 1,488 deals (seeds 9000+, GA 6) and with the same args as the stored Full-P arm
#            runs/value-p2-20261015-2044/hppo-p2 (0.629). value_p2, not P3's own critic: value_p2 has equal or better EV
#            at every turn on P3's own games (critic_compare, NOTEBOOK).
#   full_p_rep: Full-P again on the first 248 of those deals (GA 1): drift check of today's code against the stored arm
#            (win rate only: searched games are not move-identical across runs).
# Budget: credit $X, floor $X -> HOURS 7.5 at <= $X/h. VAR_CAP caps each arm (full_p3 is v3: slower per search).
cd "$(dirname "$0")/../.."
for f in data/cache/P3/best.pt data/cache/P3/metrics.jsonl data/cache/P/best.pt data/cache/value_p2/best.pt \
         data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz data/cards/lua_desc.npz \
         data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=full-p3-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
H="--honest 1 --det-tries 32 --det-fallback skip --horizon 1 --claim 1 --value data/cache/value_p2/best.pt"
GS=${GA_SEEDS:-6}; [ "${SMOKE:-0}" = 1 ] && GS=1
VARIANTS="full_p3|--opp ygoagent --ga-seeds $GS $H;full_p_rep|--opp ygoagent --ga-seeds 1 $H --ckpt data/cache/P/best.pt"
[ "${SMOKE:-0}" = 1 ] && VARIANTS="full_p3|--opp ygoagent --ga-seeds 1 --games-limit ${SMOKE_GAMES:-24} $H"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/P3/best.pt SM_ARGS='--k 4 --samples 4 --conf-skip 0.9 --deal-seed0 9000' VARIANTS='$VARIANTS' PROBE_NPROC=${PROBE_NPROC:-40} VAR_CAP=${VAR_CAP:-330m} bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-7.5}" --max-dph "${MAX_DPH:-0.38}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-20}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/P3 data/cache/value_p2 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
