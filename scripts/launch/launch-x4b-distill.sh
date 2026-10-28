#!/usr/bin/env bash
# X4b steps 3-6 (docs/plans/2026-10-28-specialists-distill.md; pre-registered NOTEBOOK 2026-10-28 "X4b"): distillation
# games, student D (per-deck teachers, pure KL), cards vs P on unrelated / related / archetype / gauntlet decks
# (scripts/vast_x4b_distill.sh on the box). Needs data/cache/x4b_spec/<arch>/{best.pt,metrics.jsonl} and
# spec_card.json from the specialist job (scripts/launch/launch-x4b-spec.sh).
# Env: SPEC_DIR (data/cache/x4b_spec), HOURS (4; 1 when SMOKE=1), MAX_DPH (0.35), SMOKE=1, ARCHS, SPEC_GAMES, ANCHOR_GAMES, EPOCHS, LR, UNREL_SEEDS,
#   REL_SEEDS, ARCH_SEEDS, GA_SEEDS, VAR_CAP (see vast_x4b_distill.sh).
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/P/metrics.jsonl data/cards/emb/qwen3-0.6b-d512.npz data/cards/lua_desc.npz \
         data/models/0546_26550M.tflite decks/x4b/split.json decks/x4b/pairs/related.pairs; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
SPEC_DIR=${SPEC_DIR:-data/cache/x4b_spec}  # SMOKE: stand-in specialists, no spec_card.json (H2 skipped)
[ "${SMOKE:-0}" = 1 ] || [ -f "$SPEC_DIR/spec_card.json" ] || { echo "missing $SPEC_DIR/spec_card.json"; exit 3; }
N=x4b-distill-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
HH=${HOURS:-4}; [ "${SMOKE:-0}" = 1 ] && HH=${HOURS:-1}
E=""
for v in ARCHS SPEC_GAMES ANCHOR_GAMES EPOCHS LR UNREL_SEEDS REL_SEEDS ARCH_SEEDS GA_SEEDS VAR_CAP; do
  [ -n "${!v:-}" ] && E="$E $v='${!v}'"
done
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} SPEC_DIR=$SPEC_DIR$E bash scripts/vast_x4b_distill.sh $N" \
  --hours "$HH" --max-dph "${MAX_DPH:-0.35}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-22}" \
  --reserve "${RESERVE:-0.10}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P "$SPEC_DIR" \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
