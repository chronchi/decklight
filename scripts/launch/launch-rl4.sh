#!/usr/bin/env bash
# RL v2 (user, 2026-10-07: "Agreed with 1-4"): PPO from base_r3 with the PFSP league (snapshots + base + ygo-agent,
# train_rl --league 1) and the privileged critic (--priv-critic 1, initialised from the rl3long critic value_ppo3).
# Same optimiser settings as rl3long (lr 6e-5, KL 0.05), so the differences are the league and the critic.
# Card checkpoints picked by mean(vs_bc, vs_bc_holdout, vs_ygoagent) (rl3long: vs_bc alone picked a base-overfit one).
# 3 h of RL + card, any GPU >=20 GB with >=24 cores (2026-10-07: no 3090 with >=24 cores listed; the 0.215 4080S had a 1.6 MB/s network; next >=24-core offers are ~0.34). Cap 4.5 h, <= $X.
cd "$(dirname "$0")/../.."
[ -f data/cache/value_ppo3/best.pt ] || { echo "missing data/cache/value_ppo3"; exit 3; }
N=rl4-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=data/cache/base_r3/best.pt RL_HOURS=3 EVAL_EVERY=40 CARD_BY=vs_bc+vs_bc_holdout+vs_ygoagent RL_ARGS='--lr 6e-5 --kl-coef 0.05 --league 1 --p-self 0.4 --p-ygoagent 0.15 --snap-every 40 --league-size 6 --priv-critic 1 --critic-init data/cache/value_ppo3/best.pt' bash scripts/vast_rl.sh $N" \
  --hours 4.5 --max-dph 0.35 --min-cpus 24 --min-gpu-ram 20 --extra-query "cpu_ram>=48" \
  --data decks data/demos data/cards/emb data/models data/cache/base_r3 data/cache/value_ppo3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
