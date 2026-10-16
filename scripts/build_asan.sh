#!/usr/bin/env bash
# AddressSanitizer build of the core + C++ env, for crash forensics only (NOTEBOOK 2026-10-15 23:30): a heap
# use-after-free in the core segfaults only for some heap layouts, so a flight replay with the release build can pass
# while ASan reports it every time. Builds into build/asan/ (core, ext); production build/ is untouched.
# Use with scripts/audit/asan_check.py. Needs scripts/build_core.sh to have applied the core patches first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/third_party/ygopro-core"
OUT="$ROOT/build/asan"
cd "$ROOT"
grep -q QUERY_CARDID "$SRC/ocgapi_constants.h" || { echo "core patches missing: run scripts/build_core.sh first"; exit 3; }
[ -f "$OUT/core/build.ninja" ] || uv run --group dev meson setup "$OUT/core" "$SRC" --buildtype=debugoptimized \
  -Ddefault_library=shared -Db_sanitize=address -Db_lundef=false > /dev/null
uv run --group dev ninja -j"${JOBS:-4}" -C "$OUT/core" > /dev/null
mkdir -p "$OUT/ext-src"
cp cpp/ygoenv/*.cpp cpp/ygoenv/*.h "$OUT/ext-src/"
sed -e "s#meson.project_source_root() / '../../third_party/ygopro-core'#'$SRC'#" \
    -e "s#meson.project_source_root() / '../../build/core'#'$OUT/core'#" cpp/ygoenv/meson.build > "$OUT/ext-src/meson.build"
[ -f "$OUT/ext/build.ninja" ] || uv run --group dev meson setup "$OUT/ext" "$OUT/ext-src" --buildtype=debugoptimized \
  -Db_sanitize=address -Db_lundef=false > /dev/null
uv run --group dev ninja -j"${JOBS:-4}" -C "$OUT/ext" > /dev/null
ls "$OUT"/ext/_ygoenv*.so
