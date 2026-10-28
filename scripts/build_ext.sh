#!/usr/bin/env bash
# Build the C++ env extension -> src/decklight/_ygoenv*.so (needs build/core from build_core.sh).
# With YGO_CORE=rust: decklight/_ygoenv_rs*.so against the Rust shim (scripts/build_rust_core.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ "${YGO_CORE:-cpp}" = rust ]; then
    OUT="$ROOT/build/ygoenv_rs"
    [ -f build/rust/libocgcore_rs.so ] || scripts/build_rust_core.sh
    if [ ! -f "$OUT/build.ninja" ]; then
        uv run --group dev meson setup "$OUT" cpp/ygoenv -Dcore=rust || { rm -rf "$OUT"; exit 1; }
    fi
    uv run --group dev ninja -C "$OUT"
    cp "$OUT"/_ygoenv_rs*.so src/decklight/
    ls -l src/decklight/_ygoenv_rs*.so
else
    OUT="$ROOT/build/ygoenv"
    [ -f build/core/libocgcore.so ] || scripts/build_core.sh
    if [ ! -f "$OUT/build.ninja" ]; then
        uv run --group dev meson setup "$OUT" cpp/ygoenv || { rm -rf "$OUT"; exit 1; }
    fi
    uv run --group dev ninja -C "$OUT"
    cp "$OUT"/_ygoenv*.so src/decklight/
    ls -l src/decklight/_ygoenv*.so
fi
