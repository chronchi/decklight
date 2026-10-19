#!/usr/bin/env bash
# Fast vs Full primer quality (user 2026-10-19: "maybe we should only use the fast mode because it's faster. Does the
# quality of the primer decrease though?"; pre-registered in NOTEBOOK 2026-10-19). 12 varied pool decks in a ring
# (data/cache/primer/pairs-fvf.txt: each vs its next 5, both seat orders = 120 ordered pairs):
#   full: the product on both seats (P + hppo k4 s4 h1 conf-skip 0.9 + value_p2), 3 seeds = 360 games (60/deck);
#   fast: raw P both seats, 15 seeds = 1,800 games (300/deck); its first 3 seeds share the full games' deals.
# Then build_primers.py --score on each dir (value_p2). Fast runs first (cheap pipeline check); the probe sizes on the
# full args (their GPU memory is the binding one). Full on CPU took 10-13 min/game/core (laptop bench 2026-10-18).
# The probe runs proc 0 of PROBE_NPROC: a Full game takes ~100 s on a GPU box (first try, 1 game per probe proc at
# PROBE_NPROC 1000, ended before the 150 s window: primer-fvf-20261019-0917), so 40 gives it ~9 games.
# Env: PROBE_NPROC (40), HOURS (3), MAX_DPH (0.45), MIN_CPUS (48), MIN_GPU_RAM (24), VAR_CAP_FULL is the shared VAR_CAP (150m).
cd "$(dirname "$0")/../.."
. scripts/product.sh
P=data/cache/primer/pairs-fvf.txt
for f in "$PRODUCT_CKPT" "$PRODUCT_VALUE" data/cards/emb/qwen3-0.6b-d512.npz $P; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=primer-fvf-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
LIM=""; [ "${SMOKE:-0}" = 1 ] && LIM="--games-limit 4"
SM="--opp self --s-seat-only 0 --deal-seed0 4000 --pairs @$P $LIM"
FULL="--seeds 3 --both-search 1 --value $PRODUCT_VALUE $PRODUCT_SEARCH"
VARIANTS="fast|--seeds 15 --raw 1;full|$FULL"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "PROBE_NPROC=${PROBE_NPROC:-40} VALUE=$PRODUCT_VALUE PROBE_ARGS='$FULL' CKPT=$PRODUCT_CKPT SM_ARGS='$SM' VARIANTS='$VARIANTS' VAR_CAP=${VAR_CAP:-150m} bash scripts/vast_primer_fvf.sh $N" \
  --hours "${HOURS:-3}" --max-dph "${MAX_DPH:-0.45}" --min-cpus "${MIN_CPUS:-48}" --min-gpu-ram "${MIN_GPU_RAM:-24}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-96}" \
  --data decks data/cards/emb data/cache/P "$PRODUCT_VALUE_DIR" data/cache/primer \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
