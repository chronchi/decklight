#!/usr/bin/env bash
# RL throughput benchmark on an RTX 4090 box (the likely RL target), base ft-swv. Cap 1 h.
cd "$(dirname "$0")/../.."
N=bench-$(date +%Y%m%d-%H%M); echo $N > data/cache/briefs/bench.tag; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r2/best.pt bash scripts/vast_bench_rl.sh $N" \
  --hours 1 --max-dph 0.40 --min-cpus 32 --min-gpu-ram 20 --extra-query "total_flops>=70 cpu_ram>=64" \
  --data decks data/cards/emb data/cache/base_r2 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
