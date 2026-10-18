#!/usr/bin/env bash
# Primer games (web-ux-plan step 5, item 7): the product's raw policy P (no search, both seats) plays every lobby
# deck (decks/pool 175 + decks/newcards 14) in a ring: each deck vs its next 5 decks in both seat orders, 1 seed,
# = 20 games per deck (1,750 + 140). The recorded hppo games only cover decks/gauntlet, which share no list with the
# lobby (muse primer smoke: coverage 0/189). Raw, not search: search overrides ~1 decision in 75 (hppo records), so
# key cards/openers/chokepoints barely change, and searched games cost ~3.5 h per 1.5k games (value-p2 card). Pairs: data/cache/primer/pairs-{pool,newcards}.txt (ring, written
# 2026-10-18). Env: HOURS (1.5), MAX_DPH (0.40), MIN_CPUS (32), VAR_CAP (40m).
cd "$(dirname "$0")/../.."
. scripts/product.sh
for f in "$PRODUCT_CKPT" "$PRODUCT_VALUE" data/cards/emb/qwen3-0.6b-d512.npz data/cache/primer/pairs-pool.txt \
         data/cache/primer/pairs-newcards.txt; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=primer-games-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
LIM=""; [ "${SMOKE:-0}" = 1 ] && LIM="--games-limit 8"
SM="--raw 1 --opp self --s-seat-only 0 --seeds 1 --deal-seed0 3000 $LIM"
VARIANTS="pool|--pairs @data/cache/primer/pairs-pool.txt;newcards|--deck-set newcards --pairs @data/cache/primer/pairs-newcards.txt"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=$PRODUCT_CKPT SM_ARGS='$SM' VARIANTS='$VARIANTS' VAR_CAP=${VAR_CAP:-40m} bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-1.5}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64}" \
  --data decks data/cards/emb data/cache/P data/cache/primer \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
