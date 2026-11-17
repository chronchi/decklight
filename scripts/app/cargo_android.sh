#!/usr/bin/env bash
# cargo for an Android ABI with the NDK's clang (ADR 0009). Usage:
#   scripts/app/cargo_android.sh <aarch64|x86_64> build --release -p duel-cli
# NDK: $ANDROID_NDK_HOME, else the newest ~/Android/Sdk/ndk/*. API level: $ANDROID_API (default 30).
set -euo pipefail
abi=$1; shift
case "$abi" in aarch64|x86_64) ;; *) echo "abi must be aarch64 or x86_64" >&2; exit 2;; esac
ndk=${ANDROID_NDK_HOME:-$(ls -d "$HOME"/Android/Sdk/ndk/* | sort -V | tail -1)}
api=${ANDROID_API:-30}
bin=$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin
target=$abi-linux-android
T=${target//-/_}
TU=$(echo "$T" | tr a-z A-Z)
export "CC_$T=$bin/$target$api-clang" "CXX_$T=$bin/$target$api-clang++" "AR_$T=$bin/llvm-ar"
export "CARGO_TARGET_${TU}_LINKER=$bin/$target$api-clang"
cd "$(dirname "$0")/../../rust"
exec taskset -c 0-3 "$HOME/.cargo/bin/cargo" "$@" --target "$target" -j 3
