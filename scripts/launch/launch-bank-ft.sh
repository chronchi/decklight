#!/usr/bin/env bash
# Point 3 diagnostic (user 2026-10-08 21:30: "If necessary after the trainings and everything continue with your point
# 3 then"; rl6 = no gain, NOTEBOOK 2026-10-09 06:40): can base_r3 learn the hint bank's winning lines, and do they
# transfer to banked starts it never trained on? rl6 gave the bank ~4.6 passes at lr 6e-5 inside PPO (hint_acc
# 0.66 -> 0.71), which does not answer it. Offline fine-tune on 75% of the banked starts (scripts/bank_ft.py), then
# every banked start played by base_r3 / ft (best held-out CE) / ftfinal vs base_r3 and rl5 it200 (1 greedy + 8 sampled
# games each), paired; then the standard report card of ft vs base_r3 (does fitting the lines hurt general play?).
cd "$(dirname "$0")/../.."
INIT=data/cache/base_r3/best.pt; BANK=data/cache/hint_gated/bank.jsonl; OPP2=data/cache/rl5_ckpts/iter-0200.pt
for f in $INIT $BANK $OPP2; do [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
N=bankft-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=$INIT BANK=$BANK OPP=$INIT OPP2=$OPP2 EPOCHS=${EPOCHS:-12} LR=${LR:-1e-4} GAMES=${GAMES:-8} CARD=${CARD:-1} bash scripts/vast_bank_ft.sh $N" \
  --hours "${HOURS:-1.5}" --max-dph "${MAX_DPH:-0.2}" --min-cpus 32 --min-gpu-ram 11 --extra-query "cpu_ram>=48" \
  --data decks data/demos data/cards/emb data/models data/cache/base_r3 data/cache/rl5_ckpts data/cache/hint_gated \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
