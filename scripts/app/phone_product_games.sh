#!/bin/sh
# A6a: run product_games in Termux on the phone (~/dl layout) and report wall + peak RSS (VmHWM).
# Copy with scp -P 8022; run as: G=<games> sh phone_product_games.sh <ort-threads>

cd ~/dl; t=$1
ORT_DYLIB_PATH=$HOME/dl/lib/libonnxruntime.so YGO_EMB=$HOME/dl/lib/qwen3-0.6b-d512.npz ./product_games --pairs decks/burning-abyss.ydk:decks/danger.ydk --games ${G:-2} --seed 5151 --choice-seed 9 --cdb-dir BabelCDB --scripts-dir CardScripts --onnx onnx --ort-threads $t --out pg_t$t.jsonl.gz --io-hash-out io_t$t.txt > pg_t$t.out 2> pg_t$t.err &
p=$!; s=$(date +%s); hwm=0
while kill -0 $p 2>/dev/null; do h=$(grep VmHWM /proc/$p/status 2>/dev/null | tr -s ' ' | cut -d' ' -f2); [ -n "$h" ] && hwm=$h; sleep 2; done
wait $p; echo "exit=$? wall=$(( $(date +%s)-s ))s vmhwm_kb=$hwm"
