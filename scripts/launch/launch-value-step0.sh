#!/usr/bin/env bash
# Value step 0 (Q of both arms at the filtered windows) on a cheap many-core box (scripts/vast_value_step0.sh).
# Env: WIN (runs/<opp-windows job>/window, required), STARTS (starts-v2.jsonl), VARIANTS (4), CAP (60m), HOURS (1.5),
#      MAX_DPH (0.35), MIN_CPUS (32).
cd "$(dirname "$0")/../.."
STARTS=${STARTS:-data/cache/line_starts/starts-v2.jsonl}
for f in "${WIN:?set WIN}" "$STARTS" data/cache/forge_r1/best.pt; do [ -e "$f" ] || { echo "missing $f"; exit 3; }; done
ls "$WIN"/*.jsonl > /dev/null || { echo "no window records in $WIN"; exit 3; }
N=value-step0-$(date +%Y%m%d-%H%M); echo $N
EXPECT=$(python3 -c "import glob,json,sys; print(sum(i['kind'] != 'agree' for f in glob.glob(sys.argv[1] + '/*.jsonl') for x in open(f) if x.strip() for i in json.loads(x)['window_info'].values()))" "$WIN")
rm -rf data/cache/v0_in && mkdir -p data/cache/v0_in && cp "$WIN"/*.jsonl data/cache/v0_in/ || { echo "copy failed"; exit 3; }
cmp <(cat "$WIN"/*.jsonl) <(cat data/cache/v0_in/*.jsonl) || { echo "copy differs"; exit 3; }
echo "labels to score: $EXPECT"; [ "$EXPECT" -gt 0 ] || exit 3
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "WIN_DIR=data/cache/v0_in EXPECT=$EXPECT STARTS=$STARTS VARIANTS=${VARIANTS:-4} CAP=${CAP:-60m} bash scripts/vast_value_step0.sh $N" \
  --hours "${HOURS:-1.5}" --max-dph "${MAX_DPH:-0.35}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 4 \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 data/cache/line_starts data/cache/v0_in \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
