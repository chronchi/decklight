#!/usr/bin/env bash
# Forge F0 (docs/plans/2026-10-29-deck-forge.md; pre-registered NOTEBOOK 2026-10-29 "Forge F0"):
# E1 iteration curve (4 new archetypes from release D, snapshots every 20), the curve card and the P-arm determinism
# rerun (scripts/vast_forge_f0.sh on the box). User 2026-10-29: "Yes to all".
# Env: HOURS (10; 1 when SMOKE=1), MAX_DPH (0.40), SMOKE=1, and the job's knobs (see vast_forge_f0.sh).
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/value_p2/best.pt data/cache/x4b_D/best.pt data/cache/x4b_D/metrics.jsonl \
         data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz data/cards/lua_desc.npz \
         decks/forge/f0/split.json decks/x4b/pairs/arch-invoked.pairs; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=forge-f0-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
HH=${HOURS:-10}; [ "${SMOKE:-0}" = 1 ] && HH=${HOURS:-1}
ENVS="SMOKE=${SMOKE:-0}"
for v in ARCHS FT_ITERS SNAP CARD_ITS FT_HOURS SHARE ARCH_GAMES MS_IT MS_MIN VAR_CAP PAR; do
  [ -n "${!v:-}" ] && ENVS="$ENVS $v='${!v}'"
done
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "$ENVS bash scripts/vast_forge_f0.sh $N" \
  --hours "$HH" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-22}" \
  --reserve "${RESERVE:-0.10}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/value_p2 data/cache/x4b_D \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
