#!/usr/bin/env bash
# Game audit of rl4 (user, 2026-10-07: "check game matchups to see if the picks/moves make sense and understand where
# it's failing"): rl4's two card checkpoints (it 80, it 144) through the exact audit-r3 card (BASE base_r2, same decks
# and seeds, every game saved + rendered), so each game can be diffed decision by decision against base_r3's
# (runs/audit-r3-20261006-1641) to see where RL changed the play.
cd "$(dirname "$0")/../.."
N=audit-rl4-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r2/best.pt CKPTS='data/cache/audit_rl4/i0080 data/cache/audit_rl4/i0144' RENDER=1 PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards --save-games 1000' bash scripts/vast_card.sh $N" \
  --hours 1 --max-dph 0.25 --min-cpus 24 --min-gpu-ram 11 --extra-query "cpu_ram>=48" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r2 data/cache/audit_rl4 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
