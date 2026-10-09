#!/usr/bin/env bash
# Search plan S0 (docs/search_plan.md): base + test-time search vs the raw base, one variant after another, each on
# every core. Processes are sized by measurement: one probe process runs 90 s, its GPU memory sets how many fit.
# Env: CKPT (base best.pt), VARIANTS ("tag|search_match args;tag2|args2"), SM_ARGS (common args), VAR_CAP (default 60m),
# PROBE_ARGS (sizing-probe variant args; default the first variant's: raw games end too fast for the 150 s probe),
# EXPECT ("tag:reference.jsonl": after variant tag, scripts/audit/harness_check.py must pass or the job stops).
# Usage: bash scripts/vast_search_match.sh <name>. Results: runs/<name>/<tag>/{games-*.jsonl,summary.json}, PROGRESS, ALERT.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_search_match.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT"
: "${CKPT:?set CKPT}"; : "${VARIANTS:?set VARIANTS}"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
[ -f "$CKPT" ] || { alert "missing $CKPT"; exit 3; }
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
J=$(nproc); [ "$Q" != max ] && J=$(( Q / P ))
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# measure one process (GPU memory, RSS) on the first variant's args, then size the pool
FIRST_ARGS="${VARIANTS%%;*}"; FIRST_ARGS="${PROBE_ARGS:-${FIRST_ARGS#*|}}"
# shellcheck disable=SC2086
uv run --no-sync python scripts/search_match.py --ckpt "$CKPT" --device cuda --out "$OUT/probe" --nproc "${PROBE_NPROC:-1000}" --proc 0 \
  ${SM_ARGS:-} $FIRST_ARGS > "$OUT/probe.txt" 2>&1 &
PP=$!; GMB=0; RMB=0
# peak over 150 s: per-process GPU memory grows after load (the 90 s single reading of s0-2021 said 256 MB; procs
# really used 270-800 MB and 17/45 hit CUDA OOM)
for i in $(seq 15); do
  sleep 10
  kill -0 $PP 2>/dev/null || { alert "probe died before 150 s: $(tail -1 "$OUT/probe.txt")"; exit 3; }
  g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -n | tail -1)
  r=$(ps -o rss= --ppid $PP 2>/dev/null | sort -n | tail -1)
  [ "${g:-0}" -gt "$GMB" ] && GMB=$g; [ $(( ${r:-0} / 1024 )) -gt "$RMB" ] && RMB=$(( ${r:-0} / 1024 ))
done
kill $PP 2>/dev/null; sleep 3; for c in $(ps -o pid= --ppid $PP 2>/dev/null); do kill "$c" 2>/dev/null; done
GTOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RTOT=$(free -m | awk '/Mem/{print $2}')
[ "$GMB" -gt 0 ] || { alert "probe: no GPU memory reading (see probe.txt)"; exit 3; }
[ "$RMB" -gt 0 ] || RMB=2000
NP=$(python3 -c "print(max(1, min($J, int($GTOT*0.85/($GMB*1.5)), int($RTOT*0.8/($RMB*1.3)))))")
echo "cores $J, gpu ${GTOT}MB, ram ${RTOT}MB; probe gpu ${GMB}MB rss ${RMB}MB per proc -> $NP procs" | tee "$OUT/sizing.txt"
IFS=';' read -ra VS <<< "$VARIANTS"
for v in "${VS[@]}"; do
  tag="${v%%|*}"; vargs="${v#*|}"; d="$OUT/$tag"; mkdir -p "$d"
  echo "$(date +%T) variant $tag: $vargs" >> "$OUT/PROGRESS"
  PIDS=()
  for ((p = 0; p < NP; p++)); do
    # shellcheck disable=SC2086
    timeout "${VAR_CAP:-60m}" uv run --no-sync python scripts/search_match.py --ckpt "$CKPT" --device cuda --out "$d" \
      --nproc "$NP" --proc "$p" ${SM_ARGS:-} $vargs > "$d/proc-$p.txt" 2>&1 &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || alert "$tag: proc pid $pid exit $?"; done
  grep -h "^ALERT\|Traceback" "$d"/proc-*.txt >> "$OUT/ALERT" 2>/dev/null || true
  if uv run --no-sync python scripts/search_match.py --summarize "$d" > "$d/summary.txt" 2>&1; then
    echo "$tag $(head -1 "$d/summary.txt")" >> "$OUT/PROGRESS"
  else
    alert "$tag: summarize failed: $(tail -1 "$d/summary.txt")"
  fi
  if [ -n "${EXPECT:-}" ] && [ "${EXPECT%%:*}" = "$tag" ]; then
    if python3 scripts/audit/harness_check.py "$d" "${EXPECT#*:}" > "$d/harness_check.txt" 2>&1; then
      echo "$tag $(head -1 "$d/harness_check.txt")" >> "$OUT/PROGRESS"
    else
      alert "$tag: $(tail -1 "$d/harness_check.txt"); later variants skipped"
      break
    fi
  fi
done
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
