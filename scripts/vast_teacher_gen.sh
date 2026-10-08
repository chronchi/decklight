#!/usr/bin/env bash
# Search-teacher data generation (PLAN §0a step 2): live searched games vs mixed opponents into
# gen_window-format records with soft targets. One variant after another, each on every core.
# Processes are sized by measurement: one probe process runs 150 s, its GPU memory sets how many fit.
# Env: CKPT (base best.pt), VARIANTS ("tag|search_match args;tag2|args2", each with --deal-mode train
# --games N --opp ...), TG_ARGS (common search args), VAR_CAP (default 300m).
# Usage: bash scripts/vast_teacher_gen.sh <name>. Results: runs/<name>/<tag>/{games-*.jsonl,summary.json},
# runs/<name>/records/<tag>/<s deck>.jsonl, PROGRESS, ALERT, LABELS (total label count).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_teacher_gen.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT"
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
FIRST_ARGS="${VARIANTS%%;*}"; FIRST_ARGS="${FIRST_ARGS#*|}"
# shellcheck disable=SC2086
uv run --no-sync python scripts/search_match.py --ckpt "$CKPT" --device cuda --out "$OUT/probe" --nproc "${PROBE_NPROC:-1000}" --proc 0 \
  --save-records "$OUT/probe-records" ${TG_ARGS:-} $FIRST_ARGS > "$OUT/probe.txt" 2>&1 &
PP=$!; GMB=0; RMB=0
# peak over 150 s: per-process GPU memory grows after load
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
count_labels() {  # $1 = records dir: "<games> games, <labels> labels"
  RD="$1" python3 -c "
import glob, json
g = n = 0
for f in glob.glob('$RD' + '/*.jsonl'):
    for line in open(f):
        if line.strip():
            g += 1
            n += sum(1 for l in json.loads(line)['labels'] if l != 'skip')
print(f'{g} games, {n} labels')"
}
IFS=';' read -ra VS <<< "$VARIANTS"
TOTAL_LABELS=0
for v in "${VS[@]}"; do
  tag="${v%%|*}"; vargs="${v#*|}"; d="$OUT/$tag"; mkdir -p "$d"
  echo "$(date +%T) variant $tag: $vargs" >> "$OUT/PROGRESS"
  PIDS=()
  for ((p = 0; p < NP; p++)); do
    # shellcheck disable=SC2086
    timeout "${VAR_CAP:-300m}" uv run --no-sync python scripts/search_match.py --ckpt "$CKPT" --device cuda --out "$d" \
      --nproc "$NP" --proc "$p" --save-records "$OUT/records/$tag" ${TG_ARGS:-} $vargs > "$d/proc-$p.txt" 2>&1 &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || alert "$tag: proc pid $pid exit $?"; done
  grep -h "^ALERT\|Traceback" "$d"/proc-*.txt >> "$OUT/ALERT" 2>/dev/null || true
  if uv run --no-sync python scripts/search_match.py --summarize "$d" > "$d/summary.txt" 2>&1; then
    echo "$tag $(head -1 "$d/summary.txt")" >> "$OUT/PROGRESS"
  else
    alert "$tag: summarize failed: $(tail -1 "$d/summary.txt")"
  fi
  CNT=$(count_labels "$OUT/records/$tag")
  NLAB=$(echo "$CNT" | grep -o '[0-9]* labels' | grep -o '[0-9]*')
  TOTAL_LABELS=$(( TOTAL_LABELS + ${NLAB:-0} ))
  echo "$tag records: $CNT" >> "$OUT/PROGRESS"
  [ "${NLAB:-0}" -gt 0 ] || alert "$tag: 0 search labels in $OUT/records/$tag"
done
echo "$TOTAL_LABELS" > "$OUT/LABELS"
echo "$(date +%T) total $TOTAL_LABELS search labels" >> "$OUT/PROGRESS"
[ "$TOTAL_LABELS" -gt 0 ] || { alert "0 search labels overall"; exit 3; }
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
