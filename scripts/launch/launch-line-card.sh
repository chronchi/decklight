#!/usr/bin/env bash
# Step-2 gate 3b on a cheap many-core box (scripts/vast_line_card.sh). Env: CAND (required), STARTS
# (starts-v2.eval.jsonl), REPEATS (4), CAP (90m), HOURS (2), MAX_DPH (0.30), MIN_CPUS (32).
cd "$(dirname "$0")/../.."
STARTS=${STARTS:-data/cache/line_starts/starts-v2.eval.jsonl}
for f in "$STARTS" "${CAND:?set CAND}" data/cache/value_p2/best.pt; do [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
N=line-card-$(date +%Y%m%d-%H%M); echo $N
mkdir -p data/cache/lc_cand && cp "$CAND" data/cache/lc_cand/cand.pt
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "STARTS=$STARTS CAND=data/cache/lc_cand/cand.pt REPEATS=${REPEATS:-4} CAP=${CAP:-90m} bash scripts/vast_line_card.sh $N" \
  --hours "${HOURS:-2}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 4 \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 data/cache/line_starts data/cache/lc_cand \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
