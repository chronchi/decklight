#!/usr/bin/env bash
# Self-imitation step 1 (user, 2026-10-07: "Agreed with the over fitting concerns, act accordingly. You can start"):
# how much better is the best of K sampled turn-1 lines than base_r3's greedy line, by end field and by rl4's
# privileged critic, and does the critic rank expert end boards above greedy's (the gate before we train on its
# picks). Train split = pool minus HOLDOUT minus kewl-tune (30 decks); holdout and ygo-agent's gauntlet decks are
# measured only (train_ok false).
cd "$(dirname "$0")/../.."
N=turn1-headroom-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt CRITIC=data/cache/critic_rl4/critic.pt HANDS=8 K=16 MAX_TRAIN=30 T1_CAP=35m bash scripts/vast_turn1.sh $N" \
  --hours 2 --max-dph 0.3 --min-cpus 32 --min-gpu-ram 11 --extra-query "cpu_ram>=48 compute_cap>=800" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r3 data/cache/critic_rl4 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
# 2026-10-07 first run (turn1-headroom-20261007-1505, T4, $X): H=20 K=32 on 40 train decks hit the 40 min cap with
# nothing written (rows were only written at the end; the T4 sat at 99% GPU). Now: per-hand tasks streamed to
# lines.partial.jsonl, PROGRESS relayed every 5 min, H=8 K=16, compute_cap>=800 (no T4).
