#!/usr/bin/env bash
# Build the Rust core shim as a shared library -> build/rust/libocgcore_rs.so
# (the OCG C API over third_party/ygopro-core-rust, see rust/ocgshim/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RS="$ROOT/third_party/ygopro-core-rust"
OUT="$ROOT/build/rust"
export PATH="$HOME/.cargo/bin:$PATH"

if ! grep -q QUERY_CARDID "$RS/ygo-core/src/constants.rs"; then
    git -C "$RS" apply "$ROOT/third_party/patches/ygopro-core-rust-query-cardid.patch"
fi
cd "$ROOT"
if [ -z "${YGO_CARDS_ONLY:-}" ] && [ "${FULL:-}" != 1 ]; then
    YGO_CARDS_ONLY="$(uv run --group dev python3 scripts/rust_card_codes.py)"
    export YGO_CARDS_ONLY
fi
export CARGO_TARGET_DIR="$OUT"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
cargo build --release --manifest-path "$ROOT/rust/ocgshim/Cargo.toml"
cp "$OUT/release/libocgcore_rs.so" "$OUT/libocgcore_rs.so"
ls -l "$OUT/libocgcore_rs.so"
