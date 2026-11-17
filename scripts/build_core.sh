#!/usr/bin/env bash
# Build edo9300 ygopro-core as a shared library -> build/core/libocgcore.so
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/third_party/ygopro-core"
OUT="$ROOT/build/core"
"$ROOT/scripts/patch_core.sh"  # Lua submodule + third_party/patches
cd "$ROOT"
if [ ! -f "$OUT/build.ninja" ]; then
    uv run --group dev meson setup "$OUT" "$SRC" --buildtype=release -Ddefault_library=shared -Db_ndebug=true
fi
uv run --group dev ninja -C "$OUT"
ls -l "$OUT"/libocgcore.so*
