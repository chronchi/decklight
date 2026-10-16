#!/usr/bin/env bash
# Preload ASan (and libstdc++: the uv Python binary does not link it, so ASan's __cxa_throw hook would find nothing)
# and run scripts/audit/asan_check.py with the given args.
cd "$(dirname "$0")/../.."
LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libstdc++.so)" ASAN_OPTIONS=detect_leaks=0 \
  exec uv run --no-sync python scripts/audit/asan_check.py "$@"
