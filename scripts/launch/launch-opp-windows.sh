#!/usr/bin/env bash
# Opponent-turn window labels on a cheap many-core box (scripts/vast_opp_windows.sh), training line starts only.
# Env: STARTS (starts-v2.jsonl), SAMPLES (16), CAP (120m), HOURS (3), MAX_DPH (0.35), MIN_CPUS (32).
cd "$(dirname "$0")/../.."
STARTS=${STARTS:-data/cache/line_starts/starts-v2.jsonl}
for f in "$STARTS" "${STARTS%.jsonl}.eval.jsonl" data/cache/forge_r1/best.pt data/cache/value_p2/best.pt; do
  [ -s "$f" ] || { echo "missing $f"; exit 3; }
done
N=opp-windows-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "STARTS=$STARTS SAMPLES=${SAMPLES:-16} CAP=${CAP:-120m} bash scripts/vast_opp_windows.sh $N" \
  --hours "${HOURS:-3}" --max-dph "${MAX_DPH:-0.35}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 4 \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 data/cache/line_starts \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
