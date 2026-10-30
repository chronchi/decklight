#!/usr/bin/env bash
# Forge round (docs/plans/2026-10-29-deck-forge.md, "Pipeline" steps 2-5): train + pack + sequentially verify
# the f2 specialists, distill the accepted packs into the next release and card it
# (scripts/vast_forge_round.sh on the box). User 2026-10-29: "Yes to all".
# Env: HOURS (8; 1 when SMOKE=1), MAX_DPH (0.30), SMOKE=1, and the job's knobs (see vast_forge_round.sh).
cd "$(dirname "$0")/../.."
for f in data/cache/x4b_D/best.pt data/cache/x4b_D/metrics.jsonl data/cache/value_p2/best.pt \
         data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz data/cards/lua_desc.npz \
         data/models/0546_26550M.tflite decks/forge/f2/split.json decks/forge/f2/pairs/reserve.pairs; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
for d in data/forge/packs/*/; do
  [ -f "$d/pack.json" ] || { echo "missing $d/pack.json"; exit 3; }
done
N=forge-round-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
HH=${HOURS:-8}; [ "${SMOKE:-0}" = 1 ] && HH=${HOURS:-1}
ENVS="SMOKE=${SMOKE:-0}"
for v in REL REL_NAME NEXT_NAME SPLIT ARCHS PACKS_IN FT_ITERS PAR SHARE FT_HOURS OPPS LOOK_GAMES K MARGIN \
         VERIFY_SEED0 TEACH_GAMES ANCHOR_GAMES EPOCHS LR CARD_SEED0 NEW_GAMES OLD_ARCHS OLD_GAMES RES_PAIRS \
         RES_GAMES GA_SEEDS VAR_CAP; do
  [ -n "${!v:-}" ] && ENVS="$ENVS $v='${!v}'"
done
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "$ENVS bash scripts/vast_forge_round.sh $N" \
  --hours "$HH" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-22}" \
  --reserve "${RESERVE:-0.10}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/value_p2 data/cache/x4b_D data/forge/packs \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
