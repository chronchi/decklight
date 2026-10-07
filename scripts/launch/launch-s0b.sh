#!/usr/bin/env bash
# Search plan v2 S0b: honest (determinized, semantic replay, 32 tries per world) vs oracle search, and the PPO value
# head (rl3long, data/cache/value_ppo3) vs the board count, all base_r3 + search vs raw base_r3, 448 games each.
# S0 references: oracle board@h1 0.696, oracle value-v2@h1 0.636. WPROCS sized by the in-script probe. Cap 6 h.
cd "$(dirname "$0")/../.."
[ -f data/cache/value_ppo3/best.pt ] || { echo "missing data/cache/value_ppo3"; exit 3; }
N=s0b-$(date +%Y%m%d-%H%M); echo $N
V="hboard|--honest 1 --det-tries 32 --horizon 1;oppo|--value data/cache/value_ppo3/best.pt --horizon 1;hppo|--honest 1 --det-tries 32 --value data/cache/value_ppo3/best.pt --horizon 1"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt SM_ARGS='--decks 8 --seeds 4 --k 4 --samples 4 --conf-skip 0.9' VARIANTS='$V' VAR_CAP=110m bash scripts/vast_search_match.sh $N" \
  --hours 6 --max-dph 0.40 --min-cpus 32 --min-gpu-ram 20 --extra-query "cpu_ram>=64" \
  --data decks data/demos data/cards/emb data/cache/base_r3 data/cache/value_ppo3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
