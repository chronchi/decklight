#!/usr/bin/env bash
# RL v0 first feel (2026-10-05): smoke then ~1.5 h PPO from bcg-20261004-2223/ft-mix; box destroyed at the end.
cd "$(dirname "$0")/../.."
N=rl-20261005-1359
RL_HOURS=1.5 taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N --cmd "RL_HOURS=1.5 bash scripts/vast_rl.sh $N" \
  --hours 2.4 --max-dph 0.2 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=48" \
  --data data/cards/emb decks data/models data/cache/rl_init > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"; tail -4 data/cache/briefs/$N.launcher.log
