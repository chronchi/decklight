#!/usr/bin/env bash
# Step-2 gate 3b (pre-registered NOTEBOOK 2026-11-04 01:40): the combo-line card on held-out lines, standalone (the
# stage-P job runs it only after a full run). `line_eval.py` compare mode (cand vs P, paired, REPEATS per start)
# sharded over the box's cores, then merged. CPU only. Env: STARTS, CAND, REPEATS (4), CAP (90m).
# Usage: bash scripts/vast_line_card.sh <name>. Do NOT run on the laptop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT/line"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export OMP_NUM_THREADS=1
uv sync --group dev --group ml -q || { alert "uv sync failed"; exit 3; }
[ -s "$CAND" ] || { alert "missing CAND $CAND"; exit 3; }
. scripts/cores.sh
J=$(box_cores)
RT=$(free -m | awk '/Mem/{print $2}'); NP=$(( J < RT * 8 / 10 / 900 ? J : RT * 8 / 10 / 900 )); [ "$NP" -lt 1 ] && NP=1
echo "cores $J, ram ${RT}MB -> $NP procs" | tee "$OUT/sizing.txt"
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
PIDS=()
for i in $(seq 0 $((NP - 1))); do
  timeout "${CAP:-90m}" uv run --no-sync python scripts/audit/line_eval.py --starts "$STARTS" --ckpt "$CAND" \
    --vs-ckpt data/cache/forge_r1/best.pt --repeats "${REPEATS:-4}" --shard "$i/$NP" --out "$OUT/line/s$i.json" \
    > "$OUT/line/s$i.txt" 2>&1 || echo "shard $i exit $?" >> "$OUT/line/fail.txt" &
  PIDS+=($!)
done
wait "${PIDS[@]}"
[ -s "$OUT/line/fail.txt" ] && alert "line_eval shards: $(sort "$OUT/line/fail.txt" | uniq -c | tr '\n' ' ' | cut -c1-200)"
uv run --no-sync python scripts/audit/line_eval.py --starts "$STARTS" --merge "$OUT"/line/s*.json \
  --out "$OUT/line/summary.json" > "$OUT/line/merge.txt" 2>&1
RC=$?
tail -1 "$OUT/line/merge.txt" | sed "s/^/line card (cand - P): /" | tee -a "$OUT/PROGRESS"
[ "$RC" -ne 0 ] && { alert "line card merge exit $RC"; exit 4; }
exit 0
