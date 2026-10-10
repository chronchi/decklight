#!/usr/bin/env bash
# ExIt pilot students (PLAN §0a "ExIt pilot"; user 2026-10-10: "Yes go ahead register decisions and start"; decisions and
# reasons in NOTEBOOK 2026-10-10 08:30). Labels: exitlab-20261009-1740 filtered by
#   scripts/exit_filter.py --min-gap 0.05 (z >= 2, <= 3/game, turns <= 2) -> data/demos/exit-g05/{ya,self,ckpt}
# flattened into data/demos/sw-xg/<opp>__<deck>.jsonl (6,321 labels). Fine-tunes base_r3 at 25/50/100% of the training
# games, then plays base_r3 and each student raw vs ygo-agent on the same 5,952 deals (2,976 per seat).
# Gates: held-out agreement with the search rises with data; going second >= +2 pp (paired, p < 0.1) for the 100%
# student; going first >= -2 pp; held-out CE not rising after epoch 1; plus a read of the rendered decisions where the
# student and base_r3 differ.
cd "$(dirname "$0")/../.."
for f in data/cache/base_r3/best.pt data/models/0546_26550M.tflite data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
NL=$(cat data/demos/sw-xg/*.jsonl | python3 -c "import sys,json;print(sum(len(json.loads(l)['search']) for l in sys.stdin if l.strip()))")
echo "sw-xg labels: $NL"; [ "$NL" -ge 5000 ] || { echo "too few labels"; exit 3; }
N=exitst-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "GA_SEEDS=${GA_SEEDS:-24} bash scripts/vast_exit_students.sh $N" \
  --hours "${HOURS:-2.5}" --max-dph "${MAX_DPH:-0.35}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 20 \
  --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data data/demos/sw-xg data/cards/emb decks data/models data/cache/base_r3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
