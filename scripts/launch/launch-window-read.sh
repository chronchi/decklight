#!/usr/bin/env bash
# Step-2 follow-up on a cheap many-core box (scripts/vast_window_read.sh): play out search's vs P's choice at every
# gate-1 window where they disagree. Env: ROWS (runs/window-gate-20261104-0735/wg), CAND (it160), STARTS
# (starts-v2.eval.jsonl), CAP (60m), HOURS (1.5), MAX_DPH (0.30), MIN_CPUS (32).
cd "$(dirname "$0")/../.."
ROWS=${ROWS:-runs/window-gate-20261104-0735/wg}; CAND=${CAND:-runs/opp-ppo-20261104-0324/rl/iter-0160.pt}
STARTS=${STARTS:-data/cache/line_starts/starts-v2.eval.jsonl}
for f in "$STARTS" "$CAND" "$ROWS/s0.json" data/cache/value_p2/best.pt; do [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
N=window-read-$(date +%Y%m%d-%H%M); echo $N
rm -rf data/cache/wr_in && mkdir -p data/cache/wr_in/rows && cp "$ROWS"/s[0-9]*.json data/cache/wr_in/rows/ && cp "$CAND" data/cache/wr_in/cand.pt
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "ROWS=data/cache/wr_in/rows STARTS=$STARTS CAND=data/cache/wr_in/cand.pt CAP=${CAP:-60m} bash scripts/vast_window_read.sh $N" \
  --hours "${HOURS:-1.5}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 4 \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 data/cache/line_starts data/cache/wr_in \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
