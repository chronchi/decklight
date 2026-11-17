#!/usr/bin/env bash
# Measure llama-server on the phone over adb (NOTEBOOK 2026-11-15 15:15). Every process this starts on the phone dies
# on its own: the server runs under the phone's `timeout` (CAP seconds), so a dropped adb connection cannot leave it
# holding the GPU and memory (2026-11-15: a nohup'd server outlived a dropped connection and slowed the user's phone),
# and the EXIT trap kills it as soon as the measurement ends. The model is deleted at the end unless KEEP=1.
#
#   scripts/phone/llm_bench.sh <serial> <llama-server binary> [prompt.json ...]
#   env: CAP (default 600 s), ARGS (extra llama-server flags), KEEP=1 keeps the model on the phone
set -euo pipefail
serial=$1; bin=$2; shift 2
adb="$HOME/Android/Sdk/platform-tools/adb -s $serial"
cap=${CAP:-600}
dir=/data/local/tmp/llmbench
model=gemma-4-E4B-it-Q4_0.gguf
rev=8903cb408c7df39cde2ed15fac397e6d7be92951
sha=a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2
size=4590807392

cleanup() {
  # -x and pidof match the process name; `pkill -f <path>` also matches (and kills) the adb shell running it.
  $adb shell "pkill -9 -x llama-server; sleep 1; pidof llama-server && echo 'ALERT: llama-server still running on the phone' || echo 'phone: no llama-server left'" || \
    echo "ALERT: adb unreachable at cleanup; the phone-side timeout kills the server within ${cap} s of its start"
  [ "${KEEP:-0}" = 1 ] || $adb shell "rm -f $dir/$model $dir/$model.part" || true
}
trap cleanup EXIT

$adb shell "mkdir -p $dir"
$adb push "$bin" $dir/llama-server >/dev/null
$adb shell "chmod 755 $dir/llama-server"
if [ "$($adb shell "stat -c %s $dir/$model 2>/dev/null || echo 0")" != "$size" ]; then
  echo "downloading $model on the phone (4.6 GB)"
  $adb shell "cd $dir && timeout 900 curl -sfL -C - -o $model.part https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/$rev/$model && mv $model.part $model"
fi
got=$($adb shell "sha256sum $dir/$model" | cut -d' ' -f1)
[ "$got" = "$sha" ] || { echo "ALERT: model sha256 $got != $sha"; exit 3; }

echo "starting llama-server (killed after ${cap} s whatever happens)"
$adb shell "cd $dir && (LD_LIBRARY_PATH=/vendor/lib64 timeout -s KILL $cap ./llama-server -m $model --jinja --host 127.0.0.1 --port 8099 ${ARGS:-} > server.log 2>&1 &)"
for _ in $(seq 120); do
  $adb shell "grep -q 'listening on' $dir/server.log" && break
  $adb shell "pidof llama-server >/dev/null" || { $adb shell "tail -20 $dir/server.log"; echo "ALERT: llama-server exited before listening"; exit 4; }
  sleep 1
done
$adb shell "grep -i -E 'opencl|adreno|offload|device|error' $dir/server.log | head -15; grep VmRSS /proc/\$(pidof llama-server)/status"
for p in "$@"; do
  $adb push "$p" $dir/prompt.json >/dev/null
  $adb shell "timeout 300 curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' -d @$dir/prompt.json" |
    python3 -c "import json,sys; r=json.load(sys.stdin); t=r['timings']; m=r['choices'][0]['message']; print('$(basename "$p")', 'prompt', t['prompt_n'], round(t['prompt_per_second'],1), 't/s', round(t['prompt_ms']/1000,1), 's; gen', t['predicted_n'], round(t['predicted_per_second'],1), 't/s; answer', repr((m.get('content') or '')[:80]))"
done
