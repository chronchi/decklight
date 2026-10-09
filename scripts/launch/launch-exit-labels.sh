#!/usr/bin/env bash
# ExIt pilot, step 1: label generation (PLAN §0a "Plan after step 3", user 2026-10-09: "Ok I approve except the scale
# up"). Search-teacher games with S going second only, searching opponent-turn-1 windows and S's own turn 2 (the
# going-second games are where raw base_r3 loses most and where E2's arms differ), each game stopped after turn 2
# (labels need no outcome). Records carry per-sample rollout scores; scripts/exit_filter.py then keeps clean
# (z >= 2), decorrelated (<= 3 per game) labels on the laptop. Opponent mix as teacher-r1 minus greedy.
# VALUE: leaf value args, the winner of P1's hppo vs hpriv (default value_ppo3).
cd "$(dirname "$0")/../.."
VALUE=${VALUE:-"--value data/cache/value_ppo3/best.pt"}
for f in data/cache/base_r3/best.pt data/cache/base_r2/best.pt data/cache/value_ppo3/best.pt data/cache/critic_rl6/critic.pt data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=exitlab-$(date +%Y%m%d-%H%M); echo $N
TOTAL=${TOTAL:-16000}  # schedule slots over both seats; --s-seat-only 1 plays half (8000 games)
NY=$(python3 -c "print(int($TOTAL*0.40))")
NS=$(python3 -c "print(int($TOTAL*0.35))")
NC=$(( TOTAL - NY - NS ))
SB=${SEED_BASE:-400000}  # teacher-r1 used 100000-130000+; fresh range
V="ya|--opp ygoagent --deal-mode train --games $NY --seed-base $SB;self|--opp self --deal-mode train --games $NS --seed-base $((SB + 20000));ckpt|--opp ckpt --opp-ckpt data/cache/base_r2/best.pt --deal-mode train --games $NC --seed-base $((SB + 40000))"
TG="--honest 1 --det-tries 32 --det-fallback skip $VALUE --k 4 --samples 4 --horizon 1 --conf-skip 0.9 --s-seat-only 1 --max-search-turn 2 --stop-turn 2"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt TG_ARGS='$TG' VARIANTS='$V' VAR_CAP=${VAR_CAP:-150m} PROBE_NPROC=30 bash scripts/vast_teacher_gen.sh $N" \
  --hours ${HOURS:-4} --max-dph ${MAX_DPH:-0.40} --min-cpus ${MIN_CPUS:-32} --min-gpu-ram 20 --extra-query "cpu_ram>=64" \
  --data decks data/cards/emb data/cache/base_r3 data/cache/base_r2 data/cache/value_ppo3 data/cache/critic_rl6 data/models > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
