#!/usr/bin/env bash
# X4b vs ygo-agent (user 2026-10-29: "how well the new D with decklists seen or related in the training and unrelated
# archetypes play against ygo-agent ... do not use search now ... raw policies"; "Sounds good, go ahead").
# Our raw policy (P, D, and each specialist on its own lists) pilots the X4b lists; ygo-agent pilots each of its 31
# gauntlet lists; both seats, own deal seeds per pair from 31000 (search_match --opp ygoagent --pairs; S lists come
# from a flattened decks/x4b). ygo-agent knows none of 62-78% of our lists' cards (vocabulary = its 864 gauntlet
# cards): absolute win rates flatter us, paired diffs do not. Analysis: scripts/audit/x4b_ga_card.py.
# Env: HOURS (1.5), MAX_DPH (0.35), ARCH_SEEDS (3: 124 pairs x 3 x 2 = 744/arm), REL_SEEDS (2: 2,976),
#   UNREL_SEEDS (3: 3,534), SMOKE=1 (--games-limit 256).
cd "$(dirname "$0")/../.."
ARCHS="branded-light-and-darkness-ritual invoked radiant-typhoon sacred-beasts atlantean-mermail"
for f in data/cache/P/best.pt data/cache/x4b_D/best.pt data/cache/x4b_D/metrics.jsonl data/models/0546_26550M.tflite \
         decks/x4b/pairs/ga-related.pairs decks/x4b/pairs/ga-unrelated.pairs; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
for a in $ARCHS; do [ -f "data/cache/x4b_spec/$a/best.pt" ] || { echo "missing specialist $a"; exit 3; }; done
N=x4b-ga-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
D=data/cache/x4b_D/best.pt
V="P-ga-related|--pairs @decks/x4b/pairs/ga-related.pairs --seeds ${REL_SEEDS:-2}"
V="$V;D-ga-related|--ckpt $D --pairs @decks/x4b/pairs/ga-related.pairs --seeds ${REL_SEEDS:-2}"
V="$V;P-ga-unrelated|--pairs @decks/x4b/pairs/ga-unrelated.pairs --seeds ${UNREL_SEEDS:-3}"
V="$V;D-ga-unrelated|--ckpt $D --pairs @decks/x4b/pairs/ga-unrelated.pairs --seeds ${UNREL_SEEDS:-3}"
for a in $ARCHS; do
  pf="--pairs @decks/x4b/pairs/ga-arch-$a.pairs --seeds ${ARCH_SEEDS:-3}"
  V="$V;P-ga-$a|$pf;D-ga-$a|--ckpt $D $pf;S-ga-$a|--ckpt data/cache/x4b_spec/$a/best.pt $pf"
done
SM="--raw 1 --opp ygoagent --seed-per-pair 1 --deal-seed0 31000 --deck-dir data/x4b_flat"
[ "${SMOKE:-0}" = 1 ] && SM="$SM --games-limit 256"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "mkdir -p data/x4b_flat && find decks/x4b -name '*.ydk' -exec cp {} data/x4b_flat/ \; && PROBE_NPROC=${PROBE_NPROC:-4} CKPT=data/cache/P/best.pt SM_ARGS='$SM' VARIANTS='$V' VAR_CAP=${VAR_CAP:-20m} bash scripts/vast_search_match.sh $N; uv run --no-sync python scripts/audit/x4b_ga_card.py runs/$N $ARCHS > runs/$N/ga_card.txt 2>&1 || echo \"ALERT x4b_ga_card: \$(tail -1 runs/$N/ga_card.txt)\" >> runs/$N/ALERT" \
  --hours "${HOURS:-1.5}" --max-dph "${MAX_DPH:-0.35}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --reserve "${RESERVE:-0.10}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-48} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/x4b_D data/cache/x4b_spec \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
