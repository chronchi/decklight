#!/usr/bin/env bash
# Hint bank (PLAN section 0a step 3): bank search-assisted successes on hard DAGS starts
# (scripts/hint_bank.py), for the RL hinted self-distillation loss (--hint-bank).
# Env: CKPT (policy best.pt, default data/cache/base_r3/best.pt), HINT_ARGS, HOURS (default 3),
# CAP (default 4.5). Cap 4.5 h at <= $X/h.
cd "$(dirname "$0")/../.."
CKPT="${CKPT:-data/cache/base_r3/best.pt}"
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 3; }
[ -s data/cache/dags/starts.jsonl ] || { echo "missing data/cache/dags/starts.jsonl (scripts/build_dags_starts.py)"; exit 3; }
HINT_ARGS="${HINT_ARGS:---n 200 --games 4 --seed 0 --k 4 --samples 4}"
N=hintbank-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=$CKPT HINT_ARGS='$HINT_ARGS' bash scripts/vast_hint_bank.sh $N" \
  --hours "${CAP:-4.5}" --max-dph 0.35 --min-cpus 24 --min-gpu-ram 20 --extra-query "cpu_ram>=48" \
  --data decks data/cards/emb data/models data/cache/base_r3 data/cache/dags > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
