#!/usr/bin/env bash
# Retrain stage D (docs/retrain/plan-v1.md §Stage D, plan-v2/v3; user green light 2026-10-11): strict re-verification of
# the stored LLM plans (guides, claude, planned, planned-s1..s4; ~15.9k lines, eval-only decks removed) plus 8-hand
# expansion of the strict records, on a many-core box (CPU work; GPU listings only, NOTEBOOK 2026-10-02 23:10).
# Only the 392 lines strict mode can change are re-verified (scripts/audit/strict_junk.py, NOTEBOOK 2026-10-11 22:15). Plan cost ~$X; expected well under.
cd "$(dirname "$0")/../.."
N=stage-d-$(date +%Y%m%d-%H%M); echo $N
D=(decks data/demos/lines data/demos/verified data/demos/decklists data/cache/audit/strict_junk.json)
for s in claude planned planned-s1 planned-s2 planned-s3 planned-s4; do D+=("data/demos/$s/lines" "data/demos/$s/verified"); done
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N --cmd "bash scripts/vast_stage_d.sh $N" \
  --hours "${HOURS:-1.5}" --max-dph "${MAX_DPH:-0.25}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 4 \
  --data "${D[@]}" > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
