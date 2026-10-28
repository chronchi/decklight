#!/usr/bin/env bash
# X4b steps 1-2 (docs/plans/2026-10-28-specialists-distill.md; pre-registered NOTEBOOK 2026-10-28 "X4b"): 5
# specialists (finetune_deck on each archetype's 3 tournament training lists) + the specialist card vs P, then the
# stop rule (scripts/vast_x4b_spec.sh on the box). User 2026-10-28: "1-3 yes, 4 let's park it. Go ahead".
# Env: HOURS (10; 0.6 when SMOKE=1), MAX_DPH (0.40), SMOKE=1 (2 iters, 2 archetypes, 64 games per arm),
#   ARCHS, FT_ITERS, FT_HOURS, PAR, ARCH_SEEDS, UNREL_SEEDS, MS_IT, MS_MIN, VAR_CAP (see vast_x4b_spec.sh).
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl data/cache/value_p2/best.pt data/cache/value_p2/metrics.jsonl \
         data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz data/cards/lua_desc.npz \
         decks/x4b/split.json decks/x4b/pairs/unrelated.pairs; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=x4b-spec-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
HH=${HOURS:-10}; ARCHS_DEF="branded-light-and-darkness-ritual invoked radiant-typhoon sacred-beasts atlantean-mermail"
if [ "${SMOKE:-0}" = 1 ]; then HH=${HOURS:-0.6}; ARCHS_DEF="invoked sacred-beasts"; fi
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} ARCHS='${ARCHS:-$ARCHS_DEF}' FT_ITERS=${FT_ITERS:-160} FT_HOURS=${FT_HOURS:-3} PAR=${PAR:-2} ARCH_SEEDS=${ARCH_SEEDS:-24} UNREL_SEEDS=${UNREL_SEEDS:-5} MS_IT=${MS_IT:-40} MS_MIN=${MS_MIN:-0.5} VAR_CAP=${VAR_CAP:-25m} bash scripts/vast_x4b_spec.sh $N" \
  --hours "$HH" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-22}" \
  --reserve "${RESERVE:-0.10}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/value_p2 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
