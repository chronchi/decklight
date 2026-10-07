#!/usr/bin/env bash
# Self-imitation step 2 training (user, 2026-10-07: "Agreed with the over fitting concerns, act accordingly. You can
# start"): base_r3 fine-tuned on its own best turn-1 lines (data/demos/selfimit/r1, train-split decks only) with a KL
# anchor (ft-si-kl1 / ft-si-kl03), then
#   1. the report card vs base_r3 (+ ygo-agent, new-card decks, combo test), and
#   2. turn-1 lines of every checkpoint (base_r3 too) on the HOLDOUT and GAUNTLET decks (never trained on): does the
#      greedy end field rise on unseen decks? (scripts/turn1_lines.py, same hands/seed for all checkpoints)
# SI=<tag> picks the data round (default r1). Shards take every data/demos/selfimit/*/*.jsonl, so exactly one round
# may be there; older rounds live in data/demos/selfimit_old/ (r2: play-out score + hand-trap veto, NOTEBOOK 2026-10-07 23:50).
cd "$(dirname "$0")/../.."
SI=${SI:-r1}
[ -d data/demos/selfimit/$SI ] || { echo "missing data/demos/selfimit/$SI"; exit 3; }
[ "$(ls data/demos/selfimit | wc -l)" = 1 ] || { echo "more than one round in data/demos/selfimit: $(ls data/demos/selfimit)"; exit 3; }
N=selfimit-ft-$SI-$(date +%Y%m%d-%H%M); echo $N
T1="for c in base:data/cache/base_r3/best.pt kl1:runs/$N/ft-si-kl1/best.pt kl03:runs/$N/ft-si-kl03/best.pt; do \
  [ -f \${c#*:} ] || continue; for s in holdout gauntlet; do \
  timeout 20m uv run --no-sync python scripts/turn1_lines.py --ckpt \${c#*:} --critic data/cache/critic_rl4/critic.pt \
    --decks \$s --hands 8 --k 4 --seed 7 --out runs/$N/t1-\${c%%:*}/\$s --procs 16 --threads 1 --device cuda \
    > runs/$N/t1-\${c%%:*}-\$s.txt 2>&1 || echo \"ALERT t1 \${c%%:*} \$s failed\" >> runs/$N/ALERT; done; done"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r3/best.pt SHARD_ONLY=selfimit VARIANTS='ft-si-kl1 ft-si-kl03' PLAY_ALL=1 SKIP_SCRATCH=1 PT_ZERO=0 PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards' bash scripts/vast_bc_gold.sh $N; $T1" \
  --hours 3 --max-dph 0.3 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r3 data/cache/critic_rl4 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
