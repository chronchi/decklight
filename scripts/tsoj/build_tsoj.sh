#!/usr/bin/env bash
# Build tsoj's black-box opponent (../yugioh_ai) with a card subset for the tsoj-arms experiment.
# Their repo has no license: this script only builds it locally; no code or data is copied into our tree.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
T="${YGO_AI:-$ROOT/../yugioh_ai}"
JOBS="${CARGO_BUILD_JOBS:-3}"
TARGET="$T/ygopro-core-rust/target"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

step() { printf '\n== %s\n' "$*"; }

[ -d "$T" ] || { echo "missing $T (clone https://github.com/tsoj/yugioh_ai there)" >&2; exit 1; }

step "submodules"
if [ ! -f "$T/ygopro-core-rust/Cargo.toml" ] || [ ! -d "$T/external/BabelCDB/.git" ] \
   || [ ! -d "$T/external/CardScripts/.git" ] || [ ! -d "$T/external/HiGHS/.git" ]; then
  git -C "$T" -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init \
    external/CardScripts external/BabelCDB external/HiGHS ygopro-core-rust
fi

step "toolchain"
command -v cargo >/dev/null || { echo "cargo not found (add ~/.cargo/bin to PATH)" >&2; exit 1; }
command -v cmake >/dev/null || { echo "cmake not found (uv tool install cmake)" >&2; exit 1; }
cargo --version; cmake --version | head -1

step "card subset"
if [ -z "${YGO_CARDS_ONLY:-}" ]; then
  YGO_CARDS_ONLY="$(cd "$ROOT" && uv run python scripts/tsoj/card_codes.py)"
fi
export YGO_CARDS_ONLY
echo "YGO_CARDS_ONLY: $(echo "$YGO_CARDS_ONLY" | wc -w) codes"

step "arbiter isready timeout: ${READY_TIMEOUT:-30} s"
# local build setting only: bots loading in parallel on a busy box need more than the default 30 s
sed -i -E "s/^const READY_TIMEOUT: f64 = [0-9.]+;/const READY_TIMEOUT: f64 = ${READY_TIMEOUT:-30}.0;/" \
  "$T/ygopro-core-rust/ygo-arbiter/src/protocol.rs"
grep -q "^const READY_TIMEOUT: f64 = ${READY_TIMEOUT:-30}.0;" "$T/ygopro-core-rust/ygo-arbiter/src/protocol.rs" \
  || { echo "READY_TIMEOUT patch did not apply" >&2; exit 1; }

step "bot"
(cd "$T" && CARGO_TARGET_DIR="$TARGET" CARGO_BUILD_JOBS="$JOBS" \
  ${TASKSET-taskset -c 0-3} cargo build --release -p bot)
step "arbiter + windbot + random bot"
(cd "$T" && CARGO_TARGET_DIR="$TARGET" CARGO_BUILD_JOBS="$JOBS" \
  ${TASKSET-taskset -c 0-3} cargo build --release --manifest-path ygopro-core-rust/Cargo.toml \
    -p ygo-arbiter --bins)
step "done"
ls -l "$TARGET/release/bot" "$TARGET/release/ygo-arbiter" "$TARGET/release/ygo-windbot" \
      "$TARGET/release/ygo-random-bot"
