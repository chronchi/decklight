#!/usr/bin/env bash
# Control for the rl5 ygo-agent drop (user 2026-10-08: "I also want to know how well the model is playing against the
# base with the gauntlet decks. Because I want to know if it's trying something crazy or not. I wonder if it's a
# matter of fine-tuning in a deck alone or what"): rl5 iterations 0/40/120/200/215 vs base_r3 (both greedy) on the
# same gauntlet schedule as launch-rl5-ga-compare.sh (31 decks, ring 4, both seats, seeds 0-2), every game saved.
# The learner never pilots gauntlet decks in training (--learner-gauntlet 0), so this separates "worse on these decks"
# from "worse against ygo-agent".
cd "$(dirname "$0")/../.."
C=data/cache/rl5_ckpts
for k in 0000 0040 0120 0200 0215; do [ -s $C/iter-$k.pt ] || { echo "missing $C/iter-$k.pt"; exit 3; }; done
[ -s data/cache/base_r3/best.pt ] || { echo "missing base_r3"; exit 3; }
N=rl5vb-$(date +%Y%m%d-%H%M); echo $N
SPECS="it0=$C/iter-0000.pt it40=$C/iter-0040.pt it120=$C/iter-0120.pt it200=$C/iter-0200.pt it215=$C/iter-0215.pt"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SEEDS=3 VS_BASE=data/cache/base_r3/best.pt bash scripts/vast_ga_compare.sh $N '$SPECS'" \
  --hours 1.0 --max-dph 0.25 --min-cpus 16 --min-gpu-ram 10 --extra-query "cpu_ram>=32" \
  --data decks data/cards/emb data/cache/base_r3 $C > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
