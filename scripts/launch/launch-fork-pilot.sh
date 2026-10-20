#!/usr/bin/env bash
# M1 fork pilot (user 2026-10-20: "Yes, agreed, fold and go ahead"; plan docs/plans/2026-10-20-rl-signal.md; gates
# pre-registered in NOTEBOOK 2026-10-20). One small GPU box, 1 h cap; credit floor $X.
cd "$(dirname "$0")/../.."
for f in data/cache/P/best.pt data/cache/value_p2/best.pt data/cards/emb/qwen3-0.6b-d512.npz data/cards/cards.cdb; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
SRC=runs/t1-boards-20261019-2309/raw
[ "$(ls $SRC/games-*.jsonl | wc -l)" -ge 1 ] || { echo "no games in $SRC"; exit 3; }
mkdir -p data/cache/fork-in/raw && cp $SRC/games-*.jsonl data/cache/fork-in/raw/
N=forkpilot-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "bash scripts/vast_fork_pilot.sh $N" \
  --hours "${HOURS:-1.0}" --max-dph "${MAX_DPH:-0.10}" --min-cpus "${MIN_CPUS:-16}" --min-gpu-ram 12 \
  --extra-query "compute_cap>=800 cpu_ram>=32" \
  --data data/cache/fork-in data/cache/P data/cache/value_p2 data/cards/emb data/cards/cards.cdb decks \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
