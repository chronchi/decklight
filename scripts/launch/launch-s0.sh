#!/usr/bin/env bash
# Search plan S0 gate: base_r3 + test-time search vs raw base_r3 (docs/search_plan.md), 448 games per variant
# (8 report-card pool decks, all ordered pairs, 4 seeds, both seats). Variants: value v2 h1 / board h1 / value v2 h2.
# Many-core GPU box (CPU-bound clone-by-replay playouts). Cap 3.5 h.
cd "$(dirname "$0")/../.."
N=s0-$(date +%Y%m%d-%H%M); echo $N
V="v2h1|--value data/cache/value_v2/best.pt --horizon 1;boardh1|--horizon 1;v2h2|--value data/cache/value_v2/best.pt --horizon 2"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt SM_ARGS='--decks 8 --seeds 4 --k 4 --samples 4 --conf-skip 0.9' VARIANTS='$V' VAR_CAP=60m bash scripts/vast_search_match.sh $N" \
  --hours 3.5 --max-dph 0.40 --min-cpus 48 --min-gpu-ram 12 --extra-query "cpu_ram>=64" \
  --data decks data/demos data/cards/emb data/cache/base_r3 data/cache/value_v2 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
