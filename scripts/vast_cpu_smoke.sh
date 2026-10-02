#!/usr/bin/env bash
# CPU-box smoke test: machine report, VecEnv throughput, and a real verification sweep timed against the laptop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
OUT="runs/$1"
mkdir -p "$OUT"
{ nproc; free -g; lscpu | grep -E "Model name|^CPU\(s\)|MHz"; cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/memory.max 2>/dev/null; df -h /; } > "$OUT/machine.txt" 2>&1
uv run python scripts/bench_vecenv.py --decks tests/fixtures/decks --envs 128 --threads 1 8 "$(nproc)" --seconds 20 > "$OUT/bench.txt" 2>&1
T0=$SECONDS
uv run python scripts/verify_lines.py --lines data/demos/lines/salamangreat.jsonl \
  --deck-from-guides data/demos/decklists --seeds 0 --budget-long 10000 --out "$OUT/verified-salamangreat.jsonl" \
  > "$OUT/verify.txt" 2>&1
echo "wall $((SECONDS - T0)) s" >> "$OUT/verify.txt"
echo done > "$OUT/DONE"
