#!/usr/bin/env bash
# Search plan v2, distillation round 3: fine-tune base_r3 on the round-2 mix + sw-b3 (oracle board@h1 window search
# labels from sw3-*), report card vs base_r3 under the loop guard. Needs data/demos/sw-b3 (scripts/relabel_window.py
# --select board@h1 --name b3 on runs/sw3-*/window). Cap 3.5 h.
cd "$(dirname "$0")/../.."
[ -d data/demos/sw-b3 ] || { echo "missing data/demos/sw-b3"; exit 3; }
N=r3-ft-r3-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/r3.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r3/best.pt VARIANTS='ft-r3' PLAY_ALL=1 SKIP_SCRATCH=1 PT_ZERO=0 PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards' bash scripts/vast_bc_gold.sh $N" \
  --hours 3.5 --max-dph 0.25 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
