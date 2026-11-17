#!/usr/bin/env bash
# Seed-resampled filter of opponent-turn window labels on a cheap many-core box (scripts/vast_filter_opp.sh).
# Env: WIN (runs/<opp-windows job>/window, required), STARTS (starts-v2.jsonl), VARIANTS (12), CAP (120m), HOURS (3),
#      MAX_DPH (0.35), MIN_CPUS (32).
cd "$(dirname "$0")/../.."
STARTS=${STARTS:-data/cache/line_starts/starts-v2.jsonl}
for f in "${WIN:?set WIN}" "$STARTS" data/cache/forge_r1/best.pt; do [ -e "$f" ] || { echo "missing $f"; exit 3; }; done
ls "$WIN"/*.jsonl > /dev/null || { echo "no window records in $WIN"; exit 3; }
N=filter-opp-$(date +%Y%m%d-%H%M); echo $N
EXPECT=$(python3 -c "import glob,json,sys; print(sum(i['kind'] != 'agree' for f in glob.glob(sys.argv[1] + '/*.jsonl') for x in open(f) if x.strip() for i in json.loads(x)['window_info'].values()))" "$WIN")
rm -rf data/cache/oppfilt_in && mkdir -p data/cache/oppfilt_in && cp "$WIN"/*.jsonl data/cache/oppfilt_in/ || { echo "copy failed"; exit 3; }
cmp <(cat "$WIN"/*.jsonl) <(cat data/cache/oppfilt_in/*.jsonl) || { echo "copy differs"; exit 3; }
echo "labels to filter: $EXPECT"; [ "$EXPECT" -gt 0 ] || exit 3
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "WIN_DIR=data/cache/oppfilt_in EXPECT=$EXPECT STARTS=$STARTS VARIANTS=${VARIANTS:-12} CAP=${CAP:-120m} bash scripts/vast_filter_opp.sh $N" \
  --hours "${HOURS:-3}" --max-dph "${MAX_DPH:-0.35}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 4 \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 data/cache/line_starts data/cache/oppfilt_in \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
