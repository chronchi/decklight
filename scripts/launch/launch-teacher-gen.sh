#!/usr/bin/env bash
# Search-teacher data generation (PLAN §0a step 2): many live searched games vs mixed opponents into
# soft-target records (train_bc --soft-targets gumbel). Default mix: ygoagent 35%, self (base_r3) 30%,
# ckpt (base_r2) 20%, greedy 15%; search args as E2. TOTAL games tunable (default 6000). Do NOT run: this
# only submits the vast job; a Monitor tails runs/<N>/{PROGRESS,ALERT,LABELS}.
cd "$(dirname "$0")/../.."
for f in data/cache/base_r3/best.pt data/cache/base_r2/best.pt data/cache/value_ppo3/best.pt data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=${NAME_PREFIX:-teacher}-$(date +%Y%m%d-%H%M); echo $N
TOTAL=${TOTAL:-6000}
NY=$(python3 -c "print(int($TOTAL*0.35))")
NS=$(python3 -c "print(int($TOTAL*0.30))")
NC=$(python3 -c "print(int($TOTAL*0.20))")
NG=$(( TOTAL - NY - NS - NC ))
# distinct seed ranges per variant and per run (SEED_BASE): teacher-probe-20261008-0710 used 100000 for all four
SB=${SEED_BASE:-100000}
V="ya|--opp ygoagent --deal-mode train --games $NY --seed-base $SB;self|--opp self --deal-mode train --games $NS --seed-base $((SB + 10000));ckpt|--opp ckpt --opp-ckpt data/cache/base_r2/best.pt --deal-mode train --games $NC --seed-base $((SB + 20000));greedy|--opp greedy --deal-mode train --games $NG --seed-base $((SB + 30000))"
TG="--honest 1 --det-tries 32 --det-fallback skip --value data/cache/value_ppo3/best.pt --k 4 --samples 4 --horizon 1 --conf-skip 0.9"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt TG_ARGS='$TG' VARIANTS='$V' VAR_CAP=${VAR_CAP:-300m} PROBE_NPROC=30 bash scripts/vast_teacher_gen.sh $N" \
  --hours ${HOURS:-8} --max-dph ${MAX_DPH:-0.40} --min-cpus ${MIN_CPUS:-32} --min-gpu-ram 20 --extra-query "cpu_ram>=64" \
  --data decks data/cards/emb data/cache/base_r3 data/cache/base_r2 data/cache/value_ppo3 data/models > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
