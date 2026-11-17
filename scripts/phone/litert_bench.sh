#!/usr/bin/env bash
# LiteRT-LM speed check on the phone over adb (NOTEBOOK 2026-11-15 15:40). Same safety rules as llm_bench.sh
# (AGENTS "The user's phone is theirs"): the run is capped on the phone (`timeout -s KILL`), killed by name at the end,
# and the 3.7 GB model is deleted unless KEEP=1.
#   scripts/phone/litert_bench.sh <serial> <backend gpu|cpu> <message.json>...   env: CAP (default 600), KEEP=1
set -euo pipefail
serial=$1; backend=$2; shift 2
adb="$HOME/Android/Sdk/platform-tools/adb -s $serial"
lit=data/cache/litert   # gpu/liblitert-lm.so = the C API zip's android_arm64 lib with SONAME set (patchelf): bionic needs it to resolve the versioned symbols
cap=${CAP:-600}
dir=/data/local/tmp/litertbench
model=gemma-4-E4B-it.litertlm
rev=f254b5ba6e36f7d89cc7892e268e9877bc20f0a0
sha=0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0
size=3659530240

cleanup() {
  $adb shell "pkill -9 -x litert_bench; sleep 1; pidof litert_bench && echo 'ALERT: litert_bench still running on the phone' || echo 'phone: no litert_bench left'" || \
    echo "ALERT: adb unreachable at cleanup; the phone-side timeout kills it within ${cap} s of its start"
  [ "${KEEP:-0}" = 1 ] || $adb shell "rm -rf $dir/$model $dir/$model.part $dir/cache" || true
}
trap cleanup EXIT

$adb shell "mkdir -p $dir/cache"
for f in $lit/litert_bench $lit/gpu/liblitert-lm.so $lit/gpu/libLiteRtGpuAccelerator.so $lit/gpu/libLiteRtOpenClAccelerator.so; do
  $adb push "$f" $dir/ >/dev/null
done
$adb shell "chmod 755 $dir/litert_bench"
if [ "$($adb shell "stat -c %s $dir/$model 2>/dev/null || echo 0")" != "$size" ]; then
  echo "downloading $model on the phone (3.7 GB)"
  $adb shell "cd $dir && timeout 900 curl -sfL -C - -o $model.part https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/$rev/$model && mv $model.part $model"
fi
got=$($adb shell "sha256sum $dir/$model" | cut -d' ' -f1)
[ "$got" = "$sha" ] || { echo "ALERT: model sha256 $got != $sha"; exit 3; }
names=""
for p in "$@"; do $adb push "$p" $dir/ >/dev/null; names="$names $(basename "$p")"; done
echo "running litert_bench on $backend (killed after ${cap} s whatever happens)"
$adb shell "cd $dir && LD_LIBRARY_PATH=$dir:/vendor/lib64 timeout -s KILL $cap ./litert_bench $model $backend $dir/cache $names 2>&1 | grep -E '^(engine|FAILED|  ttft|[^ ]*json:)'"
