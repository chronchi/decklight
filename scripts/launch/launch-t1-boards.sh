#!/usr/bin/env bash
# Is the bot's turn-1 planning the reason going first loses? (user 2026-10-19: "Let's do 1 first then it results say so
# we can plan 2"; pre-registered in NOTEBOOK 2026-10-19 "turn-1 search test"). 175 pool decks, each vs its next 4 in
# both orders (data/cache/primer/pairs-t1-r4.txt, 1,400 ordered pairs), one going-first deal per pair (seed per pair),
# S = the first player, the opponent is raw P. Same deals in every arm:
#   raw:  S raw P (the baseline; first-player rate in self-play);
#   t1h2: S searches only its own turn 1, horizon 2 (rollouts play the opponent's turn 2 in sampled worlds, so a board
#         is scored by how it survives), k4 s4, conf-skip 0.95, leaves scored by value_p2;
#   t1h1: the product search (horizon 1, conf-skip 0.9) on turn 1 only, for comparison with t1h2.
# Env: HOURS (2.5), MAX_DPH (0.40), MIN_CPUS (48), MIN_GPU_RAM (24), VAR_CAP (50m), PROBE_NPROC (40).
cd "$(dirname "$0")/../.."
. scripts/product.sh
P=data/cache/primer/pairs-t1-r4.txt
for f in "$PRODUCT_CKPT" "$PRODUCT_VALUE" data/cards/emb/qwen3-0.6b-d512.npz $P; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=t1-boards-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
LIM=""; [ "${SMOKE:-0}" = 1 ] && LIM="--games-limit 4"
SM="--opp self --s-seat-only 0 --seeds 1 --seed-per-pair 1 --deal-seed0 40000 --pairs @$P $LIM"
T1="--own-turn1 1 --max-search-turn 1 --value $PRODUCT_VALUE --honest 1 --det-tries 32 --det-fallback skip"
H2="$T1 --horizon 2 --k 4 --samples 4 --conf-skip 0.95"
H1="$T1 --horizon 1 --k 4 --samples 4 --conf-skip 0.9"
VARIANTS="raw|--raw 1;t1h2|$H2;t1h1|$H1"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "PROBE_NPROC=${PROBE_NPROC:-40} PROBE_ARGS='$H2' CKPT=$PRODUCT_CKPT SM_ARGS='$SM' VARIANTS='$VARIANTS' VAR_CAP=${VAR_CAP:-50m} bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-2.5}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-48}" --min-gpu-ram "${MIN_GPU_RAM:-24}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-96}" \
  --data decks data/cards/emb data/cache/P "$PRODUCT_VALUE_DIR" data/cache/primer \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
