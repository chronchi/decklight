#!/usr/bin/env bash
# LiteRT-LM v0.19.1 for on-device Gemma (rust/decklight-server/src/ask/ondevice.rs, NOTEBOOK 2026-11-15 15:55):
# the C API library and the GPU accelerators, pinned by sha256, into the APK's jniLibs. The release's liblitert-lm.so
# has no SONAME, so Android's linker refuses anything linked against it ("cannot find ... from verneed"); a copy gets
# one with patchelf. Everything lands in data/cache/litert/ first (not in git), then jniLibs (gitignored *.so).
set -euo pipefail
cd "$(dirname "$0")/../.."
c=data/cache/litert; jni=rust/app/src-tauri/gen/android/app/src/main/jniLibs/arm64-v8a
mkdir -p $c/gpu $jni
get() {  # url file sha256
  [ -f "$2" ] && [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$3" ] && return
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
  [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$3" ] || { echo "ALERT: $2 sha256 mismatch" >&2; exit 3; }
}
get https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.19.1/litert_lm_c_api-1.0.0.zip $c/capi.zip \
  1d430d80eaf536f696a4922270ba55eb3d112189ac198c3066d487a45e3b4d4c
media=https://media.githubusercontent.com/media/google-ai-edge/LiteRT-LM/v0.19.1/prebuilt/android_arm64
get $media/libLiteRtGpuAccelerator.so $c/gpu/libLiteRtGpuAccelerator.so 49055120bddce4f4bfb681ae1b929094ef22b238b5d9b91e8e57aeddcd58f8c8
get $media/libLiteRtOpenClAccelerator.so $c/gpu/libLiteRtOpenClAccelerator.so ef9776d87b0a4fe4793e8fa3f325dd9fecca4a2faaace14973bae6c6be2698a5
rm -rf $c/capi && mkdir $c/capi && unzip -q $c/capi.zip -d $c/capi
lib=$(find $c/capi -path '*android_arm64/liblitert-lm.so')
[ "$(sha256sum "$lib" | cut -d' ' -f1)" = d000b46697a43bfae6181f280240ddeff08ef25049902f42d54b35c519ec7764 ] || { echo "ALERT: $lib sha256 mismatch" >&2; exit 3; }
cp "$lib" $c/gpu/liblitert-lm.so && chmod u+w $c/gpu/liblitert-lm.so
uvx --from patchelf patchelf --set-soname liblitert-lm.so $c/gpu/liblitert-lm.so
cp $c/gpu/*.so $jni/
ls -la $jni
