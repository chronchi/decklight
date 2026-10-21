#!/usr/bin/env bash
# Full-P with the honest-override rule (user 2026-10-21: "Yes, let's do that", after maxx_search.py showed Full-P's
# own-turn Maxx fires are argmax-over-noise overrides). Pre-registered in NOTEBOOK 2026-10-21 "Full-P + ho card".
#   full_p_ho: P + value_p2 + the product search + --ho 1 (paired z >= 2, gap >= 0.05, 4 -> 8 samples), vs ygo-agent
#              on the SAME 1,488 deals (seeds 9000+, GA 6) and args as the stored Full-P arm
#              runs/value-p2-20261015-2044/hppo-p2 (0.629). No drift arm: full-p3-20261021-0947/full_p_rep measured
#              today's code (+2.4 pp, SE 3.0, n 248).
# Box: a 24 GB card. Procs are GPU-memory-bound (peak ~1 GB each; 14 procs at GPU_MARGIN 1.5 on full-p3's 3090 Ti),
# but 24 GB cards are ~1.25x more procs per $ than the 40+ GB offers (A100 $X, A40 $X had 13 GB disk).
# GPU_MARGIN 1.3 -> ~16 procs. Budget: credit $X, floor $X -> HOURS 7.2 at <= $X/h (<= $X).
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl data/cache/value_p2/best.pt \
         data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz data/cards/lua_desc.npz \
         data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=full-p-ho-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
H="--honest 1 --det-tries 32 --det-fallback skip --horizon 1 --claim 1 --value data/cache/value_p2/best.pt --ho 1"
VARIANTS="full_p_ho|--opp ygoagent --ga-seeds ${GA_SEEDS:-6} $H"
[ "${SMOKE:-0}" = 1 ] && VARIANTS="full_p_ho|--opp ygoagent --ga-seeds 1 --games-limit ${SMOKE_GAMES:-24} $H"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/P/best.pt SM_ARGS='--k 4 --samples 4 --conf-skip 0.9 --deal-seed0 9000' VARIANTS='$VARIANTS' PROBE_NPROC=${PROBE_NPROC:-40} VAR_CAP=${VAR_CAP:-400m} GPU_MARGIN=${GPU_MARGIN:-1.3} bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-7.2}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-22}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/value_p2 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
