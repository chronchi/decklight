#!/usr/bin/env bash
# No-loss card for X (window PPO, NOTEBOOK 2026-11-08 19:45) and A (tsoj PPO step 3): head-to-head vs P (forge_r1) on
# 16 pool decks (half holdout) x 6 seeds per ordered pair (~1,500 games, both seats), plus ygo-agent (3 seeds) and the
# new-card decks; X and A in parallel on one box (scripts/vast_card.sh). Pre-registered rule: NOTEBOOK 2026-11-08 20:30.
# data/demos is shipped: playtest samples only decks with verified lines in data/demos/*/verified (card-x-20261108-2004
# crashed without it: "need decks and num_envs > 0").
cd "$(dirname "$0")/../.."
ls data/demos/*/verified/*.jsonl > /dev/null 2>&1 || { echo "missing data/demos/*/verified"; exit 3; }
for f in data/cache/card_x/X/ckpt/best.pt data/cache/card_x/A/ckpt/best.pt data/cache/forge_r1/best.pt data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=card-x-$(date +%Y%m%d-%H%M); echo $N
PTA='--decks 16 --hands 3 --seeds 6 --narrate 0 --samples 4 --ygoagent --ga-seeds 3 --newcards decks/newcards'
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/forge_r1/best.pt CKPTS='data/cache/card_x/X data/cache/card_x/A' PT_CAP=${PT_CAP:-100m} PT_ARGS='$PTA' bash scripts/vast_card.sh $N" \
  --hours "${HOURS:-2.5}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 8 \
  --data data/demos decks data/cards/emb data/models data/cache/forge_r1 data/cache/card_x > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
