#!/usr/bin/env bash
# Forge F3: cheaper distillation (pre-registered NOTEBOOK 2026-10-30 "Forge F3"; job: scripts/vast_forge_f3.sh).
# Needs data/forge/f3/ staged from runs/forge-round-20261029-2133 (packs, anchor_decks, verify/*.pairs, kl_map.json with
# paths rewritten to data/forge/f3/packs, pgames/*-p[0-5] -> tgames/). Env: HOURS (4; smoke 1), MAX_DPH (0.30), SMOKE.
cd "$(dirname "$0")/../.."
F3=data/forge/f3
for f in data/cache/x4b_D/best.pt $F3/kl_map.json; do [ -f "$f" ] || { echo "missing $f"; exit 3; }; done
[ "$(ls $F3/packs | wc -l)" = 11 ] && [ "$(ls -d $F3/tgames/*-p* | wc -l)" = 72 ] || { echo "f3 staging incomplete"; exit 3; }
N=forge-f3-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
HH=${HOURS:-4}; [ "${SMOKE:-0}" = 1 ] && HH=${HOURS:-1}
ENVS="SMOKE=${SMOKE:-0}"
for v in ARMS EPOCHS LR D_GAMES D_WORKERS NEW_GAMES RES_GAMES OLD_SEEDS VAR_CAP; do
  [ -n "${!v:-}" ] && ENVS="$ENVS $v='${!v}'"
done
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "$ENVS bash scripts/vast_forge_f3.sh $N" \
  --hours "$HH" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-22}" \
  --reserve "${RESERVE:-0.10}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/x4b_D $F3 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
