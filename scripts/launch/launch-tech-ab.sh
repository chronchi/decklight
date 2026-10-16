#!/usr/bin/env bash
# Tech-card screen N4: 25 questions, raw P vs raw P, >= 2,000 pairs each.
# Env: HOURS (2), MAX_DPH (0.40), MIN_CPUS (32), Q, PILOT/OPP, NOPP (10), SEEDS (100),
# SEED (0), SEED_BLOCK (25), VAR_CAP (50m), PROBE_SEEDS (5).
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cards/emb/qwen3-0.6b-d512.npz docs/techcards/questions-2026-10-16.json; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=tech-ab-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "Q=${Q:-docs/techcards/questions-2026-10-16.json} PILOT=${PILOT:-data/cache/P/best.pt} OPP=${OPP:-${PILOT:-data/cache/P/best.pt}} NOPP=${NOPP:-10} SEEDS=${SEEDS:-100} SEED=${SEED:-0} SEED_BLOCK=${SEED_BLOCK:-25} VAR_CAP=${VAR_CAP:-50m} PROBE_SEEDS=${PROBE_SEEDS:-5} bash scripts/vast_tech_ab.sh $N" \
  --hours "${HOURS:-2}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-8}" \
  --extra-query "compute_cap>=${COMPUTE_CAP:-750} cpu_ram>=${MIN_RAM:-32}" \
  --data decks data/cards/emb data/models data/cache/P \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
