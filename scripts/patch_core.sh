#!/usr/bin/env bash
# Make third_party/ygopro-core buildable: its Lua submodule, then our two engine patches (third_party/patches; each
# applied once, so running this again is a no-op). Every engine build needs it: build_core.sh (Python), the app
# (rust/ygoenv-sys checks for it; rust/app/install_linux.sh runs it). Works from any cwd.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/third_party/ygopro-core"
[ -f "$SRC/ocgapi_constants.h" ] || { echo "ALERT patch_core: $SRC is empty: run 'git submodule update --init --recursive'" >&2; exit 1; }
[ -f "$SRC/lua/src/lua.h" ] || git -C "$SRC" submodule update --init --depth 1
if ! grep -q QUERY_CARDID "$SRC/ocgapi_constants.h"; then
    git -C "$SRC" apply "$ROOT/third_party/patches/ygopro-core-query-cardid.patch"
fi
# two heap-use-after-free fixes (NOTEBOOK 2026-10-15 23:30: field::reset_phase iterator, reset effects freed too early)
if ! grep -q "reset effects are no longer freed here" "$SRC/processor.cpp"; then
    git -C "$SRC" apply "$ROOT/third_party/patches/ygopro-core-uaf-fixes.patch"
fi
