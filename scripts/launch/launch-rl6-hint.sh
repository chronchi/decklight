#!/usr/bin/env bash
# RL 6 = rl5 (fixed league + Kolibri pass-rate curriculum, scripts/launch/launch-rl5-league.sh) + hinted
# self-distillation from the gated hint bank (scripts/launch/launch-hint-bank-gated.sh). One change vs rl5, so rl5 is
# its ablation. User 2026-10-08: "Agreed with your points, go ahead with this a-c before we go to the other b-d";
# "I don't want to rely completely on ygo-agent": ygo-agent stays 25% of episodes (0.5 x 0.5, as rl5) and is a
# benchmark only. The card picks checkpoints by vs base_r3 (vs_bc + vs_bc_holdout), not by vs ygo-agent.
# Kill only on collapse (NOTEBOOK 2026-10-08 17:00 rule). Env: BANK (default data/cache/hint_gated/bank.jsonl),
# HINT_COEF (default 0.5), HINT_BONUS (curriculum bonus of banked starts, default 0.25), RL_HOURS (4.5), HOURS (6.5).
# SMOKE=1: vast_rl.sh smoke mode (3 iterations, tiny eval; RL_ARGS without --iters so the smoke's own count holds),
# for the never-run hint path (rl.hints had a mask bug found 2026-10-08 23:40).
cd "$(dirname "$0")/../.."
BANK=${BANK:-data/cache/hint_gated/bank.jsonl}
for f in data/cache/value_ppo3/best.pt data/cache/base_r2/best.pt data/cache/dags/starts-train-1k.jsonl $BANK; do
  [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
N=rl6hint$([ "${SMOKE:-0}" = 1 ] && echo -smoke)-$(date +%Y%m%d-%H%M); echo $N
PTA='--decks 8 --hands 3 --seeds 8 --narrate 0 --samples 4 --ygoagent --ga-seeds 3 --newcards decks/newcards'
RLA="--lr 6e-5 --kl-coef 0.05 --league 1 --p-self 0.3 --ga-episode-share 0.5 --ga-ygo-share 0.5 --league-fixed ft-swv=data/cache/base_r2/best.pt --league-fixed-share 0.10 --league-greedy-share 0.05 --snap-every 40 --league-size 6 --priv-critic 1 --critic-init data/cache/value_ppo3/best.pt --dags-starts data/cache/dags/starts-train-1k.jsonl --dags-p 0.3 --curric 1 --ga-workers 16 $([ "${SMOKE:-0}" = 1 ] || echo --iters 400) --hint-bank $BANK --hint-coef ${HINT_COEF:-0.5} --hint-start-bonus ${HINT_BONUS:-0.25}"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=data/cache/base_r3/best.pt RL_HOURS=${RL_HOURS:-4.5} EVAL_EVERY=40 CARD_K=2 CARD_BY=vs_bc+vs_bc_holdout CARD_PT_ARGS='$PTA' RL_ARGS='$RLA' bash scripts/vast_rl.sh $N $([ "${SMOKE:-0}" = 1 ] && echo smoke || echo full)" \
  --hours "$([ "${SMOKE:-0}" = 1 ] && echo 0.7 || echo "${HOURS:-6.5}")" --max-dph 0.35 --min-cpus 32 --min-gpu-ram 20 --extra-query "cpu_ram>=48" \
  --data decks data/demos data/cards/emb data/models data/cache/base_r3 data/cache/base_r2 data/cache/value_ppo3 data/cache/dags data/cache/hint_gated \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
