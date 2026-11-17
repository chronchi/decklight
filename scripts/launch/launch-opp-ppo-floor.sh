#!/usr/bin/env bash
# IS-floor relaunch of the opponent-turn shaping PPO (NOTEBOOK 2026-11-05 08:10 pre-registration; user chose
# "Diag first, then IS floor"). Run opp-ppo-20261105-0809, RTX 3090 $X/h. HOURS 3.2 (4 refused: worst case > credit).
cd "$(dirname "$0")/../.."
STARTS=data/cache/line_starts/starts-v2.jsonl RL_EXTRA='--win-explore 0.25 --shape-opp 0.2 --isw-floor 0.5' \
  ITERS=150 MS_IT=80 MS_MIN=0.47 RL_HOURS=2.3 HOURS=3.2 bash scripts/launch/launch-opp-ppo.sh
