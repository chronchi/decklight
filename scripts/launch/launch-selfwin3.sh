#!/usr/bin/env bash
# Search plan v2, distillation labels round 3 (oracle teacher, S0's best): base_r3 self-play (league) + window search
# with the board score at horizon 1 on every unsure decision (conf-skip 0.9, own-frac 1.0), 4 samples.
# Differs from round 2 (value@h1, own-frac 0.1, no conf-skip), whose labels saturated. 26 procs sized from S0
# (518 MB GPU per proc on a 24 GB card). Cap 4 h.
cd "$(dirname "$0")/../.."
N=sw3-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/selfwin3.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r3/best.pt VALUE=data/cache/value_v2/best.pt SP_GAMES=3000 TEACHER_GAMES=0 WINDOW_GAMES=1200 WPROCS=26 SAMPLES=4 WCAP=150m WIN_SEARCH='--horizons 1 --select board@h1 --own-frac 1.0 --conf-skip 0.9' bash scripts/vast_selfwin.sh $N" \
  --hours 4 --max-dph 0.40 --min-cpus 32 --min-gpu-ram 20 --extra-query "cpu_ram>=64" \
  --data decks data/models data/cards/emb data/cache/base_r3 data/cache/value_v2 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
