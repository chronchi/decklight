#!/usr/bin/env bash
# RL v2 A/B (user, 2026-10-07: "You can launch vrpo and dags in parallel, the floor is 1usd now"): rl4's exact setup
# (base_r3, league + privileged critic, lr 6e-5, KL 0.05) plus one change: VRPO advantages (--adv vrpo: Q-critic + expected-SARSA(lambda); the privileged critic gets the Q head).
# Compare against rl4 (launch-rl4.sh) at equal iterations. 3 h of RL + card; cap 4.5 h at <= $X/h.
cd "$(dirname "$0")/../.."
[ -f data/cache/value_ppo3/best.pt ] || { echo "missing data/cache/value_ppo3"; exit 3; }
N=rl4vrpo-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=data/cache/base_r3/best.pt RL_HOURS=3 EVAL_EVERY=40 CARD_BY=vs_bc+vs_bc_holdout+vs_ygoagent RL_ARGS='--lr 6e-5 --kl-coef 0.05 --league 1 --p-self 0.4 --p-ygoagent 0.15 --snap-every 40 --league-size 6 --priv-critic 1 --critic-init data/cache/value_ppo3/best.pt --adv vrpo' bash scripts/vast_rl.sh $N" \
  --hours 4.5 --max-dph 0.35 --min-cpus 24 --min-gpu-ram 20 --extra-query "cpu_ram>=48" \
  --data decks data/demos data/cards/emb data/models data/cache/base_r3 data/cache/value_ppo3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
