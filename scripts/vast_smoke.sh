#!/usr/bin/env bash
# Phase 0 exit job: environment report, test suite, and VecEnv throughput on the instance.
set -uo pipefail
OUT="runs/$1"
mkdir -p "$OUT"
nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
lscpu > "$OUT/lscpu.txt"
uv run pytest -q tests/unit tests/engine > "$OUT/pytest.txt" 2>&1
uv run python scripts/bench_vecenv.py --decks tests/fixtures/decks --envs 128 --threads 1 4 "$(nproc)" --seconds 20 > "$OUT/bench.txt" 2>&1
echo done > "$OUT/DONE"
