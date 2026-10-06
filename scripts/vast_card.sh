#!/usr/bin/env bash
# Report card for existing checkpoints, no training: playtest each CKPT dir (ckpt/best.pt + ckpt/metrics.jsonl; the report lands in <dir>/playtest) head-to-head vs
# BASE plus the fixed opponents (PT_ARGS), all in parallel on one box with the cores split between them.
# Env: BASE (vs checkpoint), CKPTS ("dir1 dir2 ..."), PT_ARGS, PT_CAP (default 60m).
# Usage: bash scripts/vast_card.sh <name>. Results: runs/<name>/playtest-<dir>.txt, <dir>/playtest/report.json, ALERT
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_card.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT"
: "${BASE:?set BASE}"; : "${CKPTS:?set CKPTS}"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
J=$(nproc); [ "$Q" != max ] && J=$(( Q / P ))
set -- $CKPTS; N=$#; TH=$(( J / N )); [ "$TH" -lt 2 ] && TH=2
echo "cores $J, $N checkpoints, $TH threads each" > "$OUT/card.txt"
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg)"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
PIDS=()
for d in "$@"; do
  [ -f "$d/ckpt/best.pt" ] && [ -f "$d/ckpt/metrics.jsonl" ] || { alert "missing $d/ckpt/best.pt or metrics.jsonl"; continue; }
  # shellcheck disable=SC2086
  timeout "${PT_CAP:-60m}" uv run --no-sync python scripts/playtest.py --ckpt "$d/ckpt/best.pt" --vs-ckpt "$BASE" --device cuda \
    --threads "$TH" --seed 0 ${PT_ARGS:-} > "$OUT/playtest-$(basename "$d").txt" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p" || alert "playtest pid $p exit $?"; done
grep -h "^ALERT\|Traceback" "$OUT"/playtest-*.txt >> "$OUT/ALERT" 2>/dev/null || true
for d in "$@"; do mkdir -p "$OUT/$(basename "$d")"; cp -r "$d/playtest" "$OUT/$(basename "$d")/" 2>/dev/null || alert "no report for $d"; done
kill "$USAGE_PID" 2>/dev/null
echo done > "$OUT/DONE"
