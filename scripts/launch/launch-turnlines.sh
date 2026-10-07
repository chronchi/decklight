#!/usr/bin/env bash
# Whole-turn self-imitation data, round 1 (user, 2026-10-07: "Shouldn't we be also getting more lines on resource
# management/interactions with opponent in general? Not just on comboing for example? Because we don't play isolated"):
# base_r3 generates greedy + 6 sampled lines of one whole turn from 2500 real-game DAGS starts (train decks only, all
# turns, either seat) and 6 fresh turn-1 hands per train deck, the opponent responding with the policy; every line is
# scored by play-out through the opponent's next turn (8 rollouts). Selection + the hand-trap veto run on the laptop.
cd "$(dirname "$0")/../.."
N=turnlines-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt N_DAGS=2500 N_T1=6 K=6 SAMPLES=8 GEN_CAP=50m SCORE_CAP=100m bash scripts/vast_turnlines.sh $N" \
  --hours 4 --max-dph 0.3 --min-cpus 32 --min-gpu-ram 16 --extra-query "cpu_ram>=64 compute_cap>=800" \
  --data data/cards/emb decks data/models data/cache/base_r3 data/cache/dags/starts.jsonl > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
