#!/usr/bin/env bash
# Step 1a deck qualification on vast (user 2026-11-03: "You can use vast then"; the laptop run was stopped for low memory).
# Decks: data/cache/combo_pool/decks.txt (all 178 after the review fixes, NOTEBOOK 2026-11-03 22:30). One deck per process.
cd "$(dirname "$0")/../.."
for f in data/cache/forge_r1/best.pt data/cache/value_p2/best.pt data/cache/combo_pool/decks.txt; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=combo-pool-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} DECKS_FILE=${DECKS_FILE:-data/cache/combo_pool/decks.txt} MAX_NP=${MIN_CPUS:-64} bash scripts/vast_combo_pool.sh $N" \
  --hours "${HOURS:-1}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-64}" --min-compute-cap 0 \
  --extra-query "cpu_ram>=${MIN_RAM:-96} ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/cache/forge_r1 data/cache/value_p2 data/cache/combo_pool \
         data/demos/claude/verified data/demos/guides-strict-d/verified data/demos/planned/verified \
         data/demos/planned/improved/verified data/demos/planned-s1/verified data/demos/planned-s1/improved/verified \
         data/demos/planned-s2/verified data/demos/planned-s2/improved/verified data/demos/planned-s3/verified \
         data/demos/planned-s3/improved/verified data/demos/planned-s4/verified \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
