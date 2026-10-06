#!/usr/bin/env bash
# Learned value v2 (ADR 0007): outcome shards from teacher + DAgger games, value net from ft-teach-h, calibration.
cd "$(dirname "$0")/../.."
N=val-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/value.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=data/cache/base_teach_h/best.pt RECORDS='data/demos/teacher/*.jsonl data/demos/dagger/*.jsonl' STRIDE=4 EPOCHS=4 bash scripts/vast_value.sh $N" \
  --hours 1.5 --max-dph 0.15 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=32" \
  --data data/demos/teacher data/demos/dagger data/cards/emb decks data/cache/base_teach_h > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"; tail -4 data/cache/briefs/$N.launcher.log; cat runs/$N/timing.txt 2>/dev/null; grep -A12 "^calibration" runs/$N/train-value.txt 2>/dev/null
