#!/usr/bin/env bash
# Gated hint bank (NOTEBOOK 2026-10-08 21:05 quality gates, user: "We need high quality data"): candidates are rl5's
# curriculum slots (hopeless + trainable with p < 0.25), minus LP <= 1000 and >= 4 cards behind (443 slots); actor
# base_r3 (rl6's init), raw 0/4 vs base_r3 = hard; bank a hard slot only if honest search wins >= 2/4 vs base_r3 AND
# >= 1/4 vs rl5 it200 (second opponent). Slots file: runs/rl5league-20261008-1534/audit-dags/slots.jsonl copied to
# data/cache/hint_gated/ (runs/ is not pushed). Starts file = rl5's (ids index the train_rl curriculum).
# The search scores branch ends with value_ppo3, as in E2 (the search that beat ygo-agent). hintgated-20261008-2117 ran
# without VALUE_CKPT (board@h1 heuristic): 3/332 hinted wins, stopped (NOTEBOOK 2026-10-08 21:55).
cd "$(dirname "$0")/../.."
CKPT=data/cache/base_r3/best.pt; OPP2=data/cache/rl5_ckpts/iter-0200.pt; SL=data/cache/hint_gated/slots.jsonl
VAL=data/cache/value_ppo3/best.pt
# EXCLUDE=<dir with a previous run's games-*.jsonl, under data/>: extend the bank with the slots not yet run.
for f in $CKPT $OPP2 $SL $VAL data/cache/dags/starts-train-1k.jsonl; do [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
HINT_ARGS="--starts data/cache/dags/starts-train-1k.jsonl --slots $SL --n ${NSLOTS:-220} --games 4 --seed 0 --k 4 --samples 4 --min-wins 2 --opp2-min-wins 1${EXCLUDE:+ --exclude-from $EXCLUDE}"
N=hintgated-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=$CKPT OPP2_CKPT=$OPP2 VALUE_CKPT=$VAL VAR_CAP=${VAR_CAP:-150m} HINT_ARGS='$HINT_ARGS' bash scripts/vast_hint_bank.sh $N" \
  --hours "${CAP:-3.5}" --max-dph 0.35 --min-cpus 24 --min-gpu-ram 20 --extra-query "cpu_ram>=48" \
  --data decks data/cards/emb data/models data/cache/base_r3 data/cache/rl5_ckpts data/cache/dags data/cache/hint_gated data/cache/value_ppo3 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
