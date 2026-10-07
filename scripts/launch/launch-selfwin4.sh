#!/usr/bin/env bash
# Search plan v2 S2 labels, round 4: the HONEST teacher (S0b: honest + value_ppo3 0.688 vs base_r3). base_r3 self-play
# (league) + gen_window --honest 1 --det-fallback skip (no rollout ever sees the true hidden state) scored by the PPO
# critic value_ppo3 at horizon 1, on unsure decisions (conf-skip 0.9), own-frac 0.5 (honest is ~1.6x slower than
# oracle), 4 samples. WPROCS 18: the 18 sw3 workers that survived on a 24 GB card used ~1.3 GB each (26 OOMed).
# Cap 4.5 h.
cd "$(dirname "$0")/../.."
[ -f data/cache/value_ppo3/best.pt ] || { echo "missing data/cache/value_ppo3"; exit 3; }
N=sw4-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/selfwin4.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r3/best.pt VALUE=data/cache/value_ppo3/best.pt SP_GAMES=3000 TEACHER_GAMES=0 WINDOW_GAMES=1200 WPROCS=18 SAMPLES=4 WCAP=180m WIN_SEARCH='--horizons 1 --select value@h1 --own-frac 0.5 --conf-skip 0.9 --honest 1 --det-tries 32 --det-fallback skip' bash scripts/vast_selfwin.sh $N" \
  --hours 4.5 --max-dph 0.30 --min-cpus 24 --min-gpu-ram 20 --extra-query "cpu_ram>=64" \
  --data decks data/models data/cards/emb data/cache/base_r3 data/cache/value_ppo3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
