#!/usr/bin/env bash
# Build the decklight Android app (A6): arm64 APK, optimized Rust (profile.dev = release settings, see Cargo.toml),
# debuggable so run-as/logcat work on the test phone. Needs data/cache/app/apk/assets.tar.gz
# (scripts/app/build_app_assets.py) and gen/android/app/src/main/jniLibs/arm64-v8a/libonnxruntime.so
# (onnxruntime-android 1.31.1 AAR, NOTEBOOK 2026-11-02 A6a), plus LiteRT-LM for on-device Gemma
# (scripts/app/fetch_litert.sh).
set -euo pipefail
cd "$(dirname "$0")/src-tauri"
ndk=${NDK_HOME:-$(ls -d "$HOME"/Android/Sdk/ndk/* | sort -V | tail -1)}
bin=$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin
export PATH="$HOME/.cargo/bin:$PATH" ANDROID_HOME=$HOME/Android/Sdk NDK_HOME=$ndk
export JAVA_HOME=${JAVA_HOME:-$HOME/.local/jdk/jdk-21.0.8+9} CARGO_BUILD_JOBS=3
# ygoenv-sys builds the C++ engine with the NDK clang it is given (scripts/app/cargo_android.sh, API 30).
export CC_aarch64_linux_android=$bin/aarch64-linux-android30-clang CXX_aarch64_linux_android=$bin/aarch64-linux-android30-clang++
export AR_aarch64_linux_android=$bin/llvm-ar
# Always rebuild the data archive: a stale one ships old web/ files (it is deterministic, so unchanged inputs keep
# the same id and the app does not unpack again).
(cd ../../.. && uv run python scripts/app/build_app_assets.py)
test -f gen/android/app/src/main/jniLibs/arm64-v8a/libonnxruntime.so || { echo "ALERT: no libonnxruntime.so in jniLibs" >&2; exit 2; }
for so in liblitert-lm.so libLiteRtGpuAccelerator.so libLiteRtOpenClAccelerator.so; do
  test -f gen/android/app/src/main/jniLibs/arm64-v8a/$so || { echo "ALERT: no $so in jniLibs: run scripts/app/fetch_litert.sh" >&2; exit 2; }
done
rm -rf gen/android/app/build/outputs
taskset -c 0-3 npx --yes @tauri-apps/cli@^2 android build --debug --apk --target aarch64 "$@"
ls -la gen/android/app/build/outputs/apk/*/debug/*.apk
