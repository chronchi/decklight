#!/usr/bin/env bash
# Cycle-1 G4 (docs/lab/prereg/2026-11-10-master-cycle1.md): X2 and X vs tsoj's bot at 500 nodes on the 1,568 held-out
# games, paired (scripts/vast_c1_g4.sh). Stages data/cache/c1_g4 from the X export, the c1 round's X2 and X's stored
# eval. Env: HOURS (4), MAX_DPH (0.16), MIN_CPUS (32), RESERVE (0.25), SEEDS, CAP_ARM.
cd "$(dirname "$0")/../.."
G=data/cache/c1_g4; X=runs/window-ppo-20261108-0940
mkdir -p $G/X $G/X2
# metrics.jsonl next to each checkpoint: BCPolicy builds the net from its args (X2: observable history)
cp $X/train/export-X/policy/{best.pt,metrics.jsonl} $G/X/ && cp runs/c1-round-20261110-1927/X2/{best.pt,metrics.jsonl} $G/X2/ \
  && cp --update=none $X/eval-X.jsonl $G/eval-X-old.jsonl || { echo "staging $G failed"; exit 3; }
for f in $G/X/best.pt $G/X/metrics.jsonl $G/X2/best.pt $G/X2/metrics.jsonl $G/eval-X-old.jsonl data/cards/emb/qwen3-0.6b-d512.npz data/cache/x_window/best.pt data/cache/value_p2/best.pt; do
  [ -s "$f" ] || { echo "missing $f"; exit 3; }
done
N=c1-g4-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SEEDS=${SEEDS:-28} CAP_ARM=${CAP_ARM:-90m} bash scripts/vast_c1_g4.sh $N" \
  --hours "${HOURS:-4}" --reserve "${RESERVE:-0.25}" --max-dph "${MAX_DPH:-0.16}" --min-cpus "${MIN_CPUS:-32}" \
  --min-gpu-ram 4 --min-compute-cap 0 \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/x_window data/cache/value_p2 $G \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
