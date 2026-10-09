#!/usr/bin/env bash
# Point 3 (user 2026-10-08 21:30: "If necessary after the trainings and everything continue with your point 3"): the
# training target, value side. Policy distillation of the search failed 4 ways (NOTEBOOK 15:40) and RL drifts, but the
# search at play wins (+13 pp, value_ppo3 at branch ends). AlphaZero's other half: a value trained on outcomes of
# search-improved games. Data: teacher-r1 searched games (data/demos/sw-t1, 2803 games, 113.6k states at stride 4;
# S always on a pool train deck). Variants: ft = value_ppo3 fine-tuned (lr 3e-5, 3 epochs); fresh = base_r3 trunk +
# new value head (lr 1e-4, 4 epochs). value_ppo3's own score on the same val games is epoch "init" of ft.
# Gate to the search match: val MSE below value_ppo3's by a clear margin.
cd "$(dirname "$0")/../.."
for f in data/cache/value_ppo3/best.pt data/cache/base_r3/best.pt; do [ -s $f ] || { echo "missing $f"; exit 3; }; done
N=valt1-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "RECORDS='data/demos/sw-t1/*.jsonl' STRIDE=4 VARIANTS='ft:data/cache/value_ppo3/best.pt:3e-5:3 fresh:data/cache/base_r3/best.pt:1e-4:4' bash scripts/vast_value.sh $N" \
  --hours 1.2 --max-dph 0.2 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=32" \
  --data data/demos/sw-t1 data/cards/emb decks data/cache/value_ppo3 data/cache/base_r3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
