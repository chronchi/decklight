#!/usr/bin/env bash
# ADR 0007 round 1 data: self-play from the new base + multi-score window search (value v2) + ygo-agent teacher games.
# Generation only. Cap 3 h. Base staged in data/cache/base_r1/ (best.pt + metrics.jsonl).
cd "$(dirname "$0")/../.."
N=sw-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/selfwin.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r1/best.pt VALUE=data/cache/value_v2/best.pt SP_GAMES=4000 TEACHER_GAMES=6000 WINDOW_GAMES=1500 WPROCS=12 bash scripts/vast_selfwin.sh $N" \
  --hours 3 --max-dph 0.25 --min-cpus 32 --min-gpu-ram 16 --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data decks data/models data/cards/emb data/cache/base_r1 data/cache/value_v2 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"; tail -3 data/cache/briefs/$N.launcher.log; cat runs/$N/selfwin.txt runs/$N/ALERT 2>/dev/null
