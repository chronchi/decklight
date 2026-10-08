#!/usr/bin/env bash
# Self-imitation round 3: whole-turn lines with an active opponent (user, 2026-10-07: "Shouldn't we be also getting
# more lines on resource management/interactions with opponent in general? ... Because we don't play isolated").
# Data: data/demos/selfimit/r3 (turnlines-20261007-1842: 1888 real-game DAGS starts + 947 turn-1 hands, train decks,
# best of 6 samples by play-out through the opponent's next turn, rollouts clipped to ±10, hand-trap veto; NOTEBOOK
# 2026-10-08 02:00). One variant (ft-si-kl03: r1/r2 kl03 >= kl1 vs ygo-agent). Card with 4x the games (--seeds 8,
# ~900 vs base_r3: CI ~±3 pts instead of ±6.5) and base_r3 vs ygo-agent on the identical schedule; then the holdout /
# gauntlet turn-1 checks as before.
cd "$(dirname "$0")/../.."
[ -d data/demos/selfimit/r3 ] || { echo "missing data/demos/selfimit/r3"; exit 3; }
[ "$(ls data/demos/selfimit | wc -l)" = 1 ] || { echo "more than one round in data/demos/selfimit: $(ls data/demos/selfimit)"; exit 3; }
N=selfimit-ft-r3-$(date +%Y%m%d-%H%M); echo $N
PTA='--decks 8 --hands 3 --seeds 8 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards'
BASECARD="mkdir -p runs/$N/pt-base/ck && cp data/cache/base_r3/best.pt data/cache/base_r3/metrics.jsonl runs/$N/pt-base/ck/ && \
  timeout 90m uv run --no-sync python scripts/playtest.py --ckpt runs/$N/pt-base/ck/best.pt --device cuda --threads 16 --seed 0 $PTA \
  > runs/$N/playtest-base.txt 2>&1 || echo 'ALERT playtest-base failed' >> runs/$N/ALERT; rm -f runs/$N/pt-base/ck/best.pt"
T1="for c in base:data/cache/base_r3/best.pt kl03:runs/$N/ft-si-kl03/best.pt; do \
  [ -f \${c#*:} ] || continue; for s in holdout gauntlet; do \
  timeout 20m uv run --no-sync python scripts/turn1_lines.py --ckpt \${c#*:} --critic data/cache/critic_rl4/critic.pt \
    --decks \$s --hands 8 --k 4 --seed 7 --out runs/$N/t1-\${c%%:*}/\$s --procs 16 --threads 1 --device cuda \
    > runs/$N/t1-\${c%%:*}-\$s.txt 2>&1 || echo \"ALERT t1 \${c%%:*} \$s failed\" >> runs/$N/ALERT; done; done"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r3/best.pt SHARD_ONLY=selfimit VARIANTS='ft-si-kl03' PLAY_ALL=1 SKIP_SCRATCH=1 PT_ZERO=0 PT_CAP=90m PT_ARGS='$PTA' bash scripts/vast_bc_gold.sh $N; $BASECARD; $T1" \
  --hours 4 --max-dph 0.3 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r3 data/cache/critic_rl4 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
