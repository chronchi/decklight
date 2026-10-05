#!/usr/bin/env bash
# Report cards (ADR 0006) for several checkpoints in one job: each data/cache/rc/<name>/{best.pt,metrics.jsonl} gets
# the same capped playtest: vs base (data/cache/rc/zero), greedy, random, ygo-agent on its 31 lists (248 games), and
# the turn-1 combo test (8 pool decks, 4 holdout, best-of-4). Usage: bash scripts/vast_report.sh <name> [names...]
# Results: runs/<name>/playtest-<ckpt>.txt and runs/<name>/<ckpt>/playtest/report.json
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; shift; OUT="runs/$NAME"; mkdir -p "$OUT"
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  sleep 10
done
[ -f data/models/code_list.txt ] || { echo "data/models missing: ygo-agent cannot run" | tee "$OUT/ERROR"; exit 3; }
J=$(nproc); [ "$J" -gt 16 ] && J=16
CK=("$@"); [ ${#CK[@]} -eq 0 ] && CK=($(ls data/cache/rc))
run() {  # <ckpt name>
  local v="$1" d="$OUT/$1/ck"; mkdir -p "$d"; cp "data/cache/rc/$v/best.pt" "data/cache/rc/$v/metrics.jsonl" "$d/"
  VS=(); [ "$v" = zero ] || VS=(--vs-ckpt data/cache/rc/zero/best.pt)
  timeout 50m uv run --no-sync python scripts/playtest.py --ckpt "$d/best.pt" --device cuda --threads 4 --seed 0 \
    --decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent "${VS[@]}" > "$OUT/playtest-$v.txt" 2>&1
  rm -f "$d/best.pt"
}
# run the checkpoints in parallel, 4 VecEnv threads each
for v in "${CK[@]}"; do run "$v" & done
wait
for v in "${CK[@]}"; do echo "== $v"; grep -E "^vs |^combo mean|^\[" "$OUT/playtest-$v.txt"; done > "$OUT/summary.txt"
echo done > "$OUT/DONE"
