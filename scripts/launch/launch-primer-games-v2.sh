#!/usr/bin/env bash
# Lobby primer games v2 (user 2026-10-19: "Run in a vast box please"). The v1 games (launch-primer-games.sh) used one
# deal seed for every game, so a deck met ~one opening hand (modal first play 0.985; NOTEBOOK 2026-10-19). Fast
# product (raw P, both seats) is kept: not worse than Full (primer-fvf-20261019-0924). Each deck plays its next 8 ring
# decks in both seat orders (pool 175 decks: 2,800 games; newcards 14 decks, next 6: 168), one game per pair, each
# pair with its own deal seed (--seed-per-pair 1), so ~30 different hands per deck. Then build_primers --score per set.
# Env: PROBE_NPROC (1: raw games are fast), HOURS (1.5), MAX_DPH (0.40), MIN_CPUS (32), MIN_GPU_RAM (16).
cd "$(dirname "$0")/../.."
. scripts/product.sh
for f in "$PRODUCT_CKPT" "$PRODUCT_VALUE" data/cards/emb/qwen3-0.6b-d512.npz data/cache/primer/pairs-pool-r8.txt \
         data/cache/primer/pairs-newcards-r6.txt; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=primer-games-v2-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
LIM=""; [ "${SMOKE:-0}" = 1 ] && LIM="--games-limit 8"
SM="--raw 1 --opp self --s-seat-only 0 --seeds 1 --seed-per-pair 1 $LIM"
VARIANTS="pool|--deal-seed0 10000 --pairs @data/cache/primer/pairs-pool-r8.txt;newcards|--deal-seed0 20000 --deck-set newcards --pairs @data/cache/primer/pairs-newcards-r6.txt"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "PROBE_NPROC=${PROBE_NPROC:-1} VALUE=$PRODUCT_VALUE CKPT=$PRODUCT_CKPT SM_ARGS='$SM' VARIANTS='$VARIANTS' VAR_CAP=${VAR_CAP:-40m} bash scripts/vast_primer_fvf.sh $N" \
  --hours "${HOURS:-1.5}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64}" \
  --data decks data/cards/emb data/cache/P "$PRODUCT_VALUE_DIR" data/cache/primer \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
