#!/usr/bin/env bash
# Build the C++ env extension -> src/decklight/_ygoenv*.so (needs build/core from build_core.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/build/ygoenv"
cd "$ROOT"
[ -f build/core/libocgcore.so ] || scripts/build_core.sh
if [ ! -f "$OUT/build.ninja" ]; then
    uv run --group dev meson setup "$OUT" cpp/ygoenv || { rm -rf "$OUT"; exit 1; }
fi
uv run --group dev ninja -C "$OUT"
cp "$OUT"/_ygoenv*.so src/decklight/
ls -l src/decklight/_ygoenv*.so
