#!/usr/bin/env bash
# Step-2 gate 1 on a cheap many-core box (scripts/vast_window_gate.sh). Env: CAND (required: the step-2 P/best.pt,
# pulled to runs/<job>/P/best.pt), STARTS (starts-v2.eval.jsonl), SAMPLES (16), CAP (100m), HOURS (2.5),
# MAX_DPH (0.30), MIN_CPUS (32).
cd "$(dirname "$0")/../.."
STARTS=${STARTS:-data/cache/line_starts/starts-v2.eval.jsonl}
for f in "$STARTS" "${CAND:?set CAND}"; do [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
N=window-gate-$(date +%Y%m%d-%H%M); echo $N
mkdir -p data/cache/wg_cand && cp "$CAND" data/cache/wg_cand/cand.pt
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "STARTS=$STARTS CAND=data/cache/wg_cand/cand.pt SAMPLES=${SAMPLES:-16} CAP=${CAP:-100m} bash scripts/vast_window_gate.sh $N" \
  --hours "${HOURS:-2.5}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 4 \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 data/cache/line_starts data/cache/wg_cand \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
