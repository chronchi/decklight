#!/usr/bin/env bash
# Step-2 gate 1 (pre-registered NOTEBOOK 2026-11-04 01:40): scripts/audit/window_gate.py sharded over the box's cores,
# then merged. CPU only. Env: STARTS (held-out line starts), CAND (candidate checkpoint), SAMPLES (16), CAP (100m).
# Usage: bash scripts/vast_window_gate.sh <name>. Do NOT run on the laptop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT/wg"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export OMP_NUM_THREADS=1
uv sync --group dev --group ml -q || { alert "uv sync failed"; exit 3; }
uv run --no-sync python -c "import torch" || { alert "no torch"; exit 3; }
[ -s "$CAND" ] || { alert "missing CAND $CAND"; exit 3; }
. scripts/cores.sh
J=$(box_cores)
RT=$(free -m | awk '/Mem/{print $2}'); NP=$(( J < RT * 8 / 10 / 1100 ? J : RT * 8 / 10 / 1100 )); [ "$NP" -lt 1 ] && NP=1
echo "cores $J, ram ${RT}MB -> $NP procs" | tee "$OUT/sizing.txt"
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
PIDS=()
for i in $(seq 0 $((NP - 1))); do
  timeout "${CAP:-100m}" uv run --no-sync python scripts/audit/window_gate.py --starts "$STARTS" --cand "$CAND" \
    --samples "${SAMPLES:-16}" --shard "$i/$NP" --out "$OUT/wg/s$i.json" > "$OUT/wg/s$i.txt" 2>&1 \
    || echo "shard $i exit $?" >> "$OUT/wg/fail.txt" &
  PIDS+=($!)
done
wait "${PIDS[@]}"
[ -s "$OUT/wg/fail.txt" ] && alert "window_gate shards: $(sort "$OUT/wg/fail.txt" | uniq -c | tr '\n' ' ' | cut -c1-200)"
uv run --no-sync python scripts/audit/window_gate.py --starts "$STARTS" --merge "$OUT"/wg/s*.json --out "$OUT/wg/summary.json" \
  2>&1 | tail -1 | sed "s/^/gate1 /" | tee -a "$OUT/PROGRESS"
true
