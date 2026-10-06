#!/usr/bin/env bash
# ADR 0007 round 2 data: self-play from ft-swv (base_r2) under the loop guard + multi-score window search, 4 samples.
# No new teacher games (round 1: more imitation barely helped). Cap 3 h on an RTX 4090.
cd "$(dirname "$0")/../.."
N=sw2-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/selfwin2.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r2/best.pt VALUE=data/cache/value_v2/best.pt SP_GAMES=4000 TEACHER_GAMES=0 WINDOW_GAMES=1200 WPROCS=16 SAMPLES=4 WCAP=150m bash scripts/vast_selfwin.sh $N" \
  --hours 3 --max-dph 0.40 --min-cpus 32 --min-gpu-ram 16 --extra-query "total_flops>=70 cpu_ram>=64" \
  --data decks data/models data/cards/emb data/cache/base_r2 data/cache/value_v2 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
