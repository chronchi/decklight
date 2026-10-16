#!/usr/bin/env bash
# N2a scoring: product value on every decision of both seats of the value-p2 card arms
# (runs/value-p2-20261015-2044/hppo-p{,2}), then annotate_games.py --values for both arms.
# Small GPU box (RTX 3060-class or similar, >= 12 cores, max $X/h, 1 h cap).
# Env: HOURS (1), MAX_DPH (0.12), MIN_CPUS (12), MIN_GPU_RAM (10), MIN_RAM (24), SRC (the card job),
#   AUX (0/1: also score the aux heads in data/cache/aux_p2/aux.pt).
cd "$(dirname "$0")/../.."
SRC="${SRC:-runs/value-p2-20261015-2044}"
AUX="${AUX:-0}"
for f in data/cache/value_p2/best.pt data/cards/emb/qwen3-0.6b-d512.npz data/cards/cards.cdb; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
DATA="decks data/cards/emb data/cards/cards.cdb data/cache/value_p2 data/cache/score-in"
CMD="bash scripts/vast_score_decisions.sh"
if [ "$AUX" = 1 ]; then
  [ -f data/cache/aux_p2/aux.pt ] || { echo "missing data/cache/aux_p2/aux.pt (AUX=1)"; exit 3; }
  DATA="$DATA data/cache/aux_p2"
  CMD="AUX=1 bash scripts/vast_score_decisions.sh"
fi
for arm in hppo-p2 hppo-p; do
  ls "$SRC/$arm"/games-*.jsonl > /dev/null 2>&1 || { echo "no games in $SRC/$arm"; exit 3; }
done
mkdir -p data/cache/score-in
for arm in hppo-p2 hppo-p; do
  mkdir -p "data/cache/score-in/$arm"
  cp "$SRC/$arm"/games-*.jsonl "data/cache/score-in/$arm/"
done
echo "staged $(ls data/cache/score-in/hppo-p2/games-*.jsonl | wc -l) + $(ls data/cache/score-in/hppo-p/games-*.jsonl | wc -l) files"
N=score-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "$CMD $N" \
  --hours "${HOURS:-1}" --max-dph "${MAX_DPH:-0.12}" --min-cpus "${MIN_CPUS:-12}" --min-gpu-ram "${MIN_GPU_RAM:-10}" \
  --extra-query "compute_cap>=750 cpu_ram>=${MIN_RAM:-24}" \
  --data $DATA \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
