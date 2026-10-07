#!/usr/bin/env bash
# Build edo9300 ygopro-core as a shared library -> build/core/libocgcore.so
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/third_party/ygopro-core"
OUT="$ROOT/build/core"
[ -f "$SRC/lua/src/lua.h" ] || git -C "$SRC" submodule update --init --depth 1
if ! grep -q QUERY_CARDID "$SRC/ocgapi_constants.h"; then
    git -C "$SRC" apply "$ROOT/third_party/patches/ygopro-core-query-cardid.patch"
fi
cd "$ROOT"
if [ ! -f "$OUT/build.ninja" ]; then
    uv run --group dev meson setup "$OUT" "$SRC" --buildtype=release -Ddefault_library=shared -Db_ndebug=true
fi
uv run --group dev ninja -C "$OUT"
ls -l "$OUT"/libocgcore.so*
