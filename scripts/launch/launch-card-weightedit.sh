#!/usr/bin/env bash
# No-loss card for the two weight edits (NOTEBOOK 2026-11-09 16:35; user: "let's test opus combined and DeepSeek deny on
# the GPU boxes, as they both could be the new product"): head-to-head vs X (BASE) on 16 pool decks (half holdout) x 6
# seeds per ordered pair (~2,880 games, both seats), plus ygo-agent (3 seeds) and the new-card decks; both edits in
# parallel on one box (scripts/vast_card.sh), the card-x settings. Pre-registered rule (NOTEBOOK 2026-11-09 16:30):
# vs X Wilson lower bound >= 0.47 and holdout CI upper bound > 0.50.
# Weights: data/cache/card_we/<edit>/ckpt/best.pt = data/cache/weightedit/<edit>/edit.pt, metrics.jsonl = X's.
cd "$(dirname "$0")/../.."
ls data/demos/*/verified/*.jsonl > /dev/null 2>&1 || { echo "missing data/demos/*/verified"; exit 3; }
for f in data/cache/card_we/opus-combined/ckpt/best.pt data/cache/card_we/deepseek-deny/ckpt/best.pt data/cache/card_x/X/ckpt/best.pt data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
cmp -s data/cache/card_we/opus-combined/ckpt/best.pt data/cache/card_x/X/ckpt/best.pt && { echo "opus-combined == X"; exit 3; }
cmp -s data/cache/card_we/deepseek-deny/ckpt/best.pt data/cache/card_x/X/ckpt/best.pt && { echo "deepseek-deny == X"; exit 3; }
N=card-we-$(date +%Y%m%d-%H%M); echo $N
PTA='--decks 16 --hands 3 --seeds 6 --narrate 0 --samples 4 --ygoagent --ga-seeds 3 --newcards decks/newcards'
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/card_x/X/ckpt/best.pt CKPTS='data/cache/card_we/opus-combined data/cache/card_we/deepseek-deny' PT_CAP=${PT_CAP:-100m} PT_ARGS='$PTA' bash scripts/vast_card.sh $N" \
  --hours "${HOURS:-2.5}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 8 \
  --data data/demos decks data/cards/emb data/models data/cache/card_x data/cache/card_we > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
