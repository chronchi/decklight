#!/usr/bin/env bash
# Search plan v2 S2, distillation round 3b: base_r3 fine-tuned on search labels ONLY (sw-<SW>, default b3 = oracle
# board@h1) with a KL anchor to base_r3 (train_bc --kl-init 1.0 / 0.3), report card vs base_r3. Fixes ft-r3, whose
# BC mix dragged base_r3 back to BC (38.4%; NOTEBOOK 2026-10-07 03:35). Usage: [SW=b4] launch-r3kl.sh. Cap 3 h.
cd "$(dirname "$0")/../.."
SW="${SW:-b3}"
[ -d "data/demos/sw-$SW" ] || { echo "missing data/demos/sw-$SW"; exit 3; }
N=r3kl-$SW-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SW=$SW BASE=data/cache/base_r3/best.pt VARIANTS='ft-kl1 ft-kl03' PLAY_ALL=1 SKIP_SCRATCH=1 PT_ZERO=0 PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards' bash scripts/vast_bc_gold.sh $N" \
  --hours 3 --max-dph 0.25 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=64" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
