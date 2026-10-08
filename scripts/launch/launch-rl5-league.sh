#!/usr/bin/env bash
# RL 5 (PLAN §0a step 3; user 2026-10-08: "Search at play + RL (Recommended)" after the search-teacher students failed):
# fixed deck-aware league (muse league-plan option A: ygo-agent ~25% of episodes on gauntlet pairs, self 30%,
# base_r2 fixed 10%, greedy 5%, rest base_r3 + snapshots) + Kolibri pass-rate curriculum on train-deck DAGS starts
# (data/cache/dags/starts-train.jsonl, from teacher-r1 self/ckpt/greedy games; the old starts.jsonl is 95% gauntlet).
# Overfitting rule on (--learner-gauntlet 0, default): the learner never pilots a gauntlet deck. No hint term (ablation
# later). Kill at it80 if in-run vs ygo-agent < base - 2 pp. Card: top 2 by in-run vs_ygoagent on the 744-deal
# schedule, same playtest args as student-r1's base_r3 card (paired_gate.py against it).
cd "$(dirname "$0")/../.."
for f in data/cache/value_ppo3/best.pt data/cache/base_r2/best.pt data/cache/dags/starts-train.jsonl; do
  [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
N=rl5league-$(date +%Y%m%d-%H%M); echo $N
PTA='--decks 8 --hands 3 --seeds 8 --narrate 0 --samples 4 --ygoagent --ga-seeds 3 --newcards decks/newcards'
RLA='--lr 6e-5 --kl-coef 0.05 --league 1 --p-self 0.3 --ga-episode-share 0.5 --ga-ygo-share 0.5 --league-fixed ft-swv=data/cache/base_r2/best.pt --league-fixed-share 0.10 --league-greedy-share 0.05 --snap-every 40 --league-size 6 --priv-critic 1 --critic-init data/cache/value_ppo3/best.pt --dags-starts data/cache/dags/starts-train.jsonl --dags-p 0.3 --curric 1 --ga-workers 16 --iters 300'
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=data/cache/base_r3/best.pt RL_HOURS=${RL_HOURS:-4.5} EVAL_EVERY=40 CARD_K=2 CARD_BY=vs_ygoagent CARD_PT_ARGS='$PTA' RL_ARGS='$RLA' bash scripts/vast_rl.sh $N" \
  --hours "${HOURS:-6.5}" --max-dph 0.35 --min-cpus 32 --min-gpu-ram 20 --extra-query "cpu_ram>=48" \
  --data decks data/demos data/cards/emb data/models data/cache/base_r3 data/cache/base_r2 data/cache/value_ppo3 data/cache/dags \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
