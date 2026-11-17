#!/usr/bin/env bash
# Continuation-bias check on a cheap many-core box (scripts/vast_line_bias.sh). Env: STARTS (held-out line starts,
# default starts-v2.eval.jsonl), CAP (90m), HOURS (2.5), MAX_DPH (0.30), MIN_CPUS (64).
cd "$(dirname "$0")/../.."
STARTS=${STARTS:-data/cache/line_starts/starts-v2.eval.jsonl}
[ -s "$STARTS" ] || { echo "missing $STARTS"; exit 3; }
N=line-bias-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "STARTS=$STARTS CAP=${CAP:-90m} bash scripts/vast_line_bias.sh $N" \
  --hours "${HOURS:-2.5}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-64}" --min-gpu-ram 4 \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 data/cache/line_starts \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
