#!/usr/bin/env bash
# Hint bank (PLAN section 0a step 3): bank search-assisted successes on hard DAGS starts.
# Processes are sized by measurement: one probe proc runs 150 s, its GPU memory sets how many fit.
# Env: CKPT (policy best.pt), HINT_ARGS (common hint_bank.py args: --n/--games/--seed/--k/...),
# OPP_CKPT (default: CKPT), VALUE_CKPT (default: unset), VAR_CAP (default 120m).
# Usage: CKPT=... HINT_ARGS='--n 200 --games 4 ...' bash scripts/vast_hint_bank.sh <name>.
# Results: runs/<name>/bank/{bank.jsonl,summary.json}, PROGRESS, ALERT.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_hint_bank.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT"
: "${CKPT:?set CKPT}"; : "${HINT_ARGS:?set HINT_ARGS}"
OPP_CKPT="${OPP_CKPT:-$CKPT}"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
[ -f "$CKPT" ] || { alert "missing $CKPT"; exit 3; }
[ -s data/cache/dags/starts.jsonl ] || { alert "missing data/cache/dags/starts.jsonl"; exit 3; }
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
J=$(nproc); [ "$Q" != max ] && J=$(( Q / P ))
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D="$OUT/bank"; mkdir -p "$D"
# measure one process (GPU memory, RSS) on a small probe, then size the pool
# shellcheck disable=SC2086
uv run --no-sync python scripts/hint_bank.py --ckpt "$CKPT" --opp-ckpt "$OPP_CKPT" ${VALUE_CKPT:+--value "$VALUE_CKPT"} \
  --device cuda --out "$OUT/probe" --nproc "${PROBE_NPROC:-1000}" --proc 0 --n 4 --games 1 \
  --k 2 --samples 1 > "$OUT/probe.txt" 2>&1 &
PP=$!; GMB=0; RMB=0
for i in $(seq 15); do
  sleep 10
  kill -0 $PP 2>/dev/null || break # a tiny probe may finish before 150 s: size from what we measured
  g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -n | tail -1)
  r=$(ps -o rss= --ppid $PP 2>/dev/null | sort -n | tail -1)
  [ "${g:-0}" -gt "$GMB" ] && GMB=$g; [ $(( ${r:-0} / 1024 )) -gt "$RMB" ] && RMB=$(( ${r:-0} / 1024 ))
done
kill $PP 2>/dev/null; sleep 3; for c in $(ps -o pid= --ppid $PP 2>/dev/null); do kill "$c" 2>/dev/null; done
wait $PP 2>/dev/null && echo "probe finished early (banked or failed on 4 starts): see probe.txt" | tee -a "$OUT/PROGRESS"
GTOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RTOT=$(free -m | awk '/Mem/{print $2}')
[ "$GMB" -gt 0 ] || { alert "probe: no GPU memory reading (see probe.txt)"; exit 3; }
[ "$RMB" -gt 0 ] || RMB=2000
NP=$(python3 -c "print(max(1, min($J, int($GTOT*0.85/($GMB*1.5)), int($RTOT*0.8/($RMB*1.3)))))")
echo "cores $J, gpu ${GTOT}MB, ram ${RTOT}MB; probe gpu ${GMB}MB rss ${RMB}MB per proc -> $NP procs" | tee "$OUT/sizing.txt"
echo "$(date +%T) bank: $NP procs, args $HINT_ARGS" >> "$OUT/PROGRESS"
PIDS=()
for ((p = 0; p < NP; p++)); do
  # shellcheck disable=SC2086
  timeout "${VAR_CAP:-120m}" uv run --no-sync python scripts/hint_bank.py --ckpt "$CKPT" --opp-ckpt "$OPP_CKPT" \
    ${VALUE_CKPT:+--value "$VALUE_CKPT"} --device cuda --out "$D" --nproc "$NP" --proc "$p" \
    $HINT_ARGS > "$D/proc-$p.txt" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || alert "bank: proc pid $pid exit $?"; done
grep -h "^ALERT\|Traceback" "$D"/proc-*.txt >> "$OUT/ALERT" 2>/dev/null || true
if uv run --no-sync python scripts/hint_bank.py --summarize "$D" > "$D/summary.txt" 2>&1; then
  head -1 "$D/summary.txt" >> "$OUT/PROGRESS"
else
  alert "bank: summarize failed: $(tail -1 "$D/summary.txt")"
fi
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
