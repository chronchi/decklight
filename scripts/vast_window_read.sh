#!/usr/bin/env bash
# Step-2 follow-up (NOTEBOOK 2026-11-04 14:30): scripts/audit/window_read.py sharded over the box's cores (both
# choices played out at every gate-1 window where search and P disagree, plus controls), then merged. CPU only.
# Env: ROWS (gate-1 rows dir), STARTS, CAND, CAP (60m). Usage: bash scripts/vast_window_read.sh <name>. Not on the laptop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT/wr"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export OMP_NUM_THREADS=1
uv sync --group dev --group ml -q || { alert "uv sync failed"; exit 3; }
[ -s "$CAND" ] || { alert "missing CAND $CAND"; exit 3; }
. scripts/cores.sh
J=$(box_cores)
RT=$(free -m | awk '/Mem/{print $2}'); NP=$(( J < RT * 8 / 10 / 1100 ? J : RT * 8 / 10 / 1100 )); [ "$NP" -lt 1 ] && NP=1
echo "cores $J, ram ${RT}MB -> $NP procs" | tee "$OUT/sizing.txt"
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
PIDS=()
for i in $(seq 0 $((NP - 1))); do
  timeout "${CAP:-60m}" uv run --no-sync python scripts/audit/window_read.py --rows "$ROWS" --starts "$STARTS" \
    --cand "$CAND" --text-n 4 --shard "$i/$NP" --out "$OUT/wr/s$i.json" > "$OUT/wr/s$i.log" 2>&1 \
    || echo "shard $i exit $?" >> "$OUT/wr/fail.txt" &
  PIDS+=($!)
done
wait "${PIDS[@]}"
[ -s "$OUT/wr/fail.txt" ] && alert "window_read shards: $(sort "$OUT/wr/fail.txt" | uniq -c | tr '\n' ' ' | cut -c1-200)"
uv run --no-sync python scripts/audit/window_read.py --merge "$OUT"/wr/s*.json --out "$OUT/wr/summary.json" \
  > "$OUT/wr/merge.txt" 2>&1
RC=$?
tail -1 "$OUT/wr/merge.txt" | sed "s/^/window_read /" | tee -a "$OUT/PROGRESS"
[ "$RC" -ne 0 ] && { alert "window_read merge exit $RC"; exit 4; }
exit 0
