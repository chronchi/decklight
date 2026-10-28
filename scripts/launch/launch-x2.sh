#!/usr/bin/env bash
# X2 diagnostic (docs/plans/2026-10-27-experiments-and-rust.md; pre-registered NOTEBOOK 2026-10-27 "X2"): does P lose
# strength on fresh tournament lists of archetypes it trained on? User 2026-10-27: "Yes" (X1, X2, R0-R2 scope).
#   t: P raw pilots the archetype's TRAINING list vs 8 fixed train decks (decks/variants/x2-T.pairs);
#   v: P raw pilots the FRESH list (decks/variants/x2-V.pairs, decks/variants/*.ydk from scripts/x2_variants.py).
# Opponent raw P, both seats, SEEDS deal seeds per pair from 15000 (--seed-per-pair: pair k of t and of v share seeds).
# Env: HOURS (1), MAX_DPH (0.35), SEEDS (8 -> 216 pairs x 8 x 2 = 3,456 games per arm), SMOKE=1 (--games-limit 64).
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cards/emb/qwen3-0.6b-d512.npz decks/variants/x2-T.pairs decks/variants/x2-V.pairs; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=x2-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
SM="--raw 1 --opp ckpt --opp-ckpt data/cache/P/best.pt --seeds ${SEEDS:-8} --seed-per-pair 1 --deal-seed0 15000 --deck-dir decks/variants"
[ "${SMOKE:-0}" = 1 ] && SM="$SM --games-limit 64"
VARIANTS="t|--pairs @decks/variants/x2-T.pairs;v|--pairs @decks/variants/x2-V.pairs"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "PROBE_NPROC=${PROBE_NPROC:-40} CKPT=data/cache/P/best.pt SM_ARGS='$SM' VARIANTS='$VARIANTS' VAR_CAP=${VAR_CAP:-25m} bash scripts/vast_search_match.sh $N && uv run --no-sync python scripts/audit/x2_paired.py runs/$N/t runs/$N/v > runs/$N/paired.txt 2>&1" \
  --hours "${HOURS:-1}" --max-dph "${MAX_DPH:-0.35}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-48} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cache/P \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
