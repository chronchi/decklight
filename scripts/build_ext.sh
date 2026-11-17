#!/usr/bin/env bash
# Build the C++ env extension -> src/decklight/_ygoenv*.so (needs build/core from build_core.sh).
# With YGO_CORE=rust: decklight/_ygoenv_rs*.so against the Rust shim (scripts/build_rust_core.sh).
set -euo pipefail
# Install by rename, never cp over the old file: a running bot has it mmapped, and overwriting the same inode
# segfaults it when it first touches a replaced page (two long jobs exited -11, NOTEBOOK 2026-11-10 14:23).
install_so() { local f; for f in "$@"; do cp "$f" "src/decklight/.$(basename "$f").tmp" && mv -f "src/decklight/.$(basename "$f").tmp" "src/decklight/$(basename "$f")"; done; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ "${YGO_CORE:-cpp}" = rust ]; then
    OUT="$ROOT/build/ygoenv_rs"
    [ -f build/rust/libocgcore_rs.so ] || scripts/build_rust_core.sh
    if [ ! -f "$OUT/build.ninja" ]; then
        uv run --group dev meson setup "$OUT" cpp/ygoenv -Dcore=rust || { rm -rf "$OUT"; exit 1; }
    fi
    uv run --group dev ninja -C "$OUT"
    install_so "$OUT"/_ygoenv_rs*.so
    ls -l src/decklight/_ygoenv_rs*.so
else
    OUT="$ROOT/build/ygoenv"
    [ -f build/core/libocgcore.so ] || scripts/build_core.sh
    if [ ! -f "$OUT/build.ninja" ]; then
        uv run --group dev meson setup "$OUT" cpp/ygoenv || { rm -rf "$OUT"; exit 1; }
    fi
    uv run --group dev ninja -C "$OUT"
    install_so "$OUT"/_ygoenv*.so
    ls -l src/decklight/_ygoenv*.so
fi
