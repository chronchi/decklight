#!/usr/bin/env bash
# Why does rl5 lose to ygo-agent? (user 2026-10-08: "I want to first understand the hands played, for example which
# games it was winning before and why is it losing now? Did it forget some combos or is it because it's misplaying
# more"). Replays the in-run vs-ygo-agent eval schedule (32 gauntlet decks, ring 4, both seats) for 3 seeds with
# iterations 0/40/120/200 of rl5league-20261008-1534, saving every game; seed 0 must reproduce the in-run eval.
cd "$(dirname "$0")/../.."
C=data/cache/rl5_ckpts
for k in 0000 0040 0120 0200; do [ -s $C/iter-$k.pt ] || { echo "missing $C/iter-$k.pt"; exit 3; }; done
N=rl5ga-$(date +%Y%m%d-%H%M); echo $N
SPECS="it0=$C/iter-0000.pt:0.419 it40=$C/iter-0040.pt:0.399 it120=$C/iter-0120.pt:0.387 it200=$C/iter-0200.pt:0.315"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SEEDS=3 bash scripts/vast_ga_compare.sh $N '$SPECS'" \
  --hours 1.2 --max-dph 0.25 --min-cpus 32 --min-gpu-ram 10 --extra-query "cpu_ram>=48" \
  --data decks data/cards/emb data/models $C > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
