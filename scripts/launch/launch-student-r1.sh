#!/usr/bin/env bash
# Search-teacher student, round 1 (PLAN §0a step 2; NOTEBOOK 2026-10-08). Data: data/demos/sw-t1 = teacher-probe-20261008-0710
# + teacher-r1-20261008-0838 records (base_r3 + honest search, E2 settings, vs ygo-agent 35% / self 30% / base_r2 20% /
# greedy 15%; our side on pool train decks only — user: "Our side on pool decks (Recommended)"). Fine-tune base_r3 on
# soft targets softmax(prior + c*q) of the searched decisions (c = 5 and 10; median selection-score spread 0.077, so c=5
# moves logits ~0.4). Card: vs ygo-agent on the 744-deal schedule (31 gauntlet lists x next 4 x 3 seeds x 2 seats, the E2
# deals) + ~900 games vs base_r3; base_r3 carded on the identical schedule. Gate: scripts/audit/paired_gate.py base vs
# variant report.json (--match ygoagent).
cd "$(dirname "$0")/../.."
[ -d data/demos/sw-t1 ] || { echo "missing data/demos/sw-t1"; exit 3; }
NL=$(uv run --no-sync python -c "import json,glob;print(sum(len(json.loads(l).get('search') or []) for f in glob.glob('data/demos/sw-t1/*.jsonl') for l in open(f) if l.strip()))")
echo "sw-t1 labels: $NL"; [ "$NL" -ge "${MIN_LABELS:-50000}" ] || { echo "too few labels ($NL)"; exit 3; }
N=student-r1-$(date +%Y%m%d-%H%M); echo $N
PTA='--decks 8 --hands 3 --seeds 8 --narrate 0 --samples 4 --ygoagent --ga-seeds 3 --newcards decks/newcards'
BASECARD="mkdir -p runs/$N/pt-base/ck && cp data/cache/base_r3/best.pt data/cache/base_r3/metrics.jsonl runs/$N/pt-base/ck/ && \
  timeout 150m uv run --no-sync python scripts/playtest.py --ckpt runs/$N/pt-base/ck/best.pt --device cuda --threads 16 --seed 0 $PTA \
  > runs/$N/playtest-base.txt 2>&1 || echo 'ALERT playtest-base failed' >> runs/$N/ALERT; rm -f runs/$N/pt-base/ck/best.pt"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r3/best.pt ST=t1 SHARD_ONLY=sw VARIANTS='ft-st-c5 ft-st-c10' PLAY_ALL=1 SKIP_SCRATCH=1 PT_ZERO=0 PT_CAP=150m PT_ARGS='$PTA' bash scripts/vast_bc_gold.sh $N; $BASECARD" \
  --hours "${HOURS:-4}" --max-dph 0.3 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
