#!/usr/bin/env bash
# Turn-1 combo check (user 2026-10-11: "did you check the combos made by base r3 + search compared to our data used for
# behavior cloning? … And what about for unseen decks?"): base_r3 greedy and best-of-16 vs the verified expert lines on
# the SAME hands, on train, holdout and new-card decks. New-card expert lines are muse-planned, engine-verified,
# eval-only (data/eval_lines/newcards, written by plan_lines.py --deck-dir decks/newcards). Same recipe as
# turn1-headroom-20261007-1611 (H=8 then; 12 now = every new-card verified hand). hppo never searches turn 1, so the
# product's turn-1 play is exactly base_r3 greedy.
cd "$(dirname "$0")/../.."
for f in data/cache/base_r3/best.pt data/cache/critic_rl4/critic.pt; do [ -f "$f" ] || { echo "missing $f"; exit 3; }; done
N=combo-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt CRITIC=data/cache/critic_rl4/critic.pt HANDS=12 K=16 MAX_TRAIN=30 SPLITS='train holdout newcards' T1_CAP=35m bash scripts/vast_turn1.sh $N" \
  --hours 2 --max-dph 0.3 --min-cpus 32 --min-gpu-ram 11 --extra-query "cpu_ram>=48 compute_cap>=800" \
  --data data/demos data/eval_lines data/cards/emb decks data/models data/cache/base_r3 data/cache/critic_rl4 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
