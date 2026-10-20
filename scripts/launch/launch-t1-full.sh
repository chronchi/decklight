#!/usr/bin/env bash
# Does turn-1 search add to decklight's Full bot? (user 2026-10-20: "Yes, let's run 1"; pre-registered in NOTEBOOK
# 2026-10-20 "Full + turn-1 search test"). Same 1,400 going-first deals as t1-boards-20261019-2309
# (data/cache/primer/pairs-t1-r4.txt, deal-seed0 40000, seed per pair); S = first player, opponent raw P:
#   full:    S = the product search as decklight runs it (turns >= 2 + windows; own turn 1 unsearched);
#   full_t1: the same plus its own turn 1 searched (--own-turn1 1).
# Budget: credit $X, floor $X (user 2026-10-20: "It's ok if it goes to X of vast credit") -> HOURS 2.6 at <=
# $X/h (<= $X); each arm capped (VAR_CAP 60m), paired on shared deals.
cd "$(dirname "$0")/../.."
. scripts/product.sh
P=data/cache/primer/pairs-t1-r4.txt
for f in "$PRODUCT_CKPT" "$PRODUCT_VALUE" data/cards/emb/qwen3-0.6b-d512.npz $P; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=t1-full-$(date +%Y%m%d-%H%M); echo $N
SM="--opp self --s-seat-only 0 --seeds 1 --seed-per-pair 1 --deal-seed0 40000 --pairs @$P"
FULL="--value $PRODUCT_VALUE $PRODUCT_SEARCH"
VARIANTS="full|$FULL;full_t1|$FULL --own-turn1 1"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "PROBE_NPROC=${PROBE_NPROC:-40} PROBE_ARGS='$FULL --own-turn1 1' CKPT=$PRODUCT_CKPT SM_ARGS='$SM' VARIANTS='$VARIANTS' VAR_CAP=${VAR_CAP:-60m} bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-2.6}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-48}" --min-gpu-ram "${MIN_GPU_RAM:-24}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-96}" \
  --data decks data/cards/emb data/cache/P "$PRODUCT_VALUE_DIR" data/cache/primer \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
