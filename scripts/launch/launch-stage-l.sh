#!/usr/bin/env bash
# Retrain stage L (docs/retrain/plan-v2.md §Stage L, plan-v3; user green light 2026-10-11): honest decisive window labels
# (hw) on base_r3 self-play over pool train decks, labelled by base_r3 + value_ppo3 in decisive-label mode (s4 -> s12),
# then a replication audit (>= 0.85 sign holds at 32 fresh worlds, else the labels are not used).
# Plan cost ~$X, ~1.5 h of search on a many-core box; capped at WCAP of search + ~40 min self-play/probe/replication.
# Launch only after stage S's replication of ho overrides is >= 0.85 (same decisive rule). 2026-10-12: at z=2 it was
# 0.752 (n=246); overrides decided at >= 3 SE held 0.888 (n=98), >= 4 SE 0.939 (n=49), so L runs at HO_Z=3 and its own
# replication audit (0.85) still decides whether the labels are used (NOTEBOOK 2026-10-12 02:15).
cd "$(dirname "$0")/../.."
for f in data/cache/base_r3/best.pt data/cache/value_ppo3/best.pt; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=stage-l-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "HO_Z=${HO_Z:-3} WCAP=${WCAP:-100m} WG=${WG:-1500} SP_GAMES=${SP_GAMES:-3000} bash scripts/vast_stage_l.sh $N" \
  --hours "${HOURS:-3.5}" --max-dph "${MAX_DPH:-0.32}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-24}" \
  --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data decks data/cards/emb data/cache/base_r3 data/cache/value_ppo3 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
