#!/usr/bin/env bash
# Retrain stage M play-out anchor (docs/retrain/plan-v3.md §Stage M): scripts/turn_anchor.py over every stage M turn
# record (data/demos/turns), base_r3 + value_ppo3, 8 samples per arm, horizon 2. Processes sized from a 120 s probe
# (min(cores, GPU memory, RAM)); records split by line-id hash. Fails loud on a bucket kept-share < 0.3 (plan: drop and
# investigate that bucket) or 0 records.
# Env: SAMPLES (8), HORIZON (2), CAP (90m). Usage: bash scripts/vast_turn_anchor.sh <name>. Vast GPU only.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT/anchor"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
BASE=data/cache/base_r3/best.pt; VALUE=data/cache/value_ppo3/best.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv sync --group dev --group ml -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
J=$(nproc); [ "$Q" != max ] && J=$(( Q / P ))
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
A=(--turns 'data/demos/turns/*/*.jsonl' --ckpt "$BASE" --value "$VALUE" --device cuda --samples "${SAMPLES:-8}"
   --horizon "${HORIZON:-2}")
uv run --no-sync python scripts/turn_anchor.py "${A[@]}" --proc 0 --nproc 20 --out "$OUT/probe.jsonl" > "$OUT/probe.txt" 2>&1 &
PP=$!; GMB=0; RMB=0
for i in $(seq 12); do
  sleep 10; kill -0 $PP 2>/dev/null || break
  g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -n | tail -1)
  r=$(ps -o rss= --ppid $PP 2>/dev/null | sort -n | tail -1)
  [ "${g:-0}" -gt "$GMB" ] && GMB=$g; [ $(( ${r:-0} / 1024 )) -gt "$RMB" ] && RMB=$(( ${r:-0} / 1024 ))
done
kill $PP 2>/dev/null; sleep 3; for c in $(ps -o pid= --ppid $PP 2>/dev/null); do kill "$c" 2>/dev/null; done
[ "$GMB" -gt 0 ] || { alert "probe: no GPU memory reading: $(tail -1 "$OUT/probe.txt")"; exit 3; }
[ "$RMB" -gt 0 ] || RMB=2000
GTOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); RTOT=$(free -m | awk '/Mem/{print $2}')
NP=$(python3 -c "print(max(1, min($J, int($GTOT*0.85/($GMB*1.3)), int($RTOT*0.8/($RMB*1.3)))))")
echo "cores $J, gpu ${GTOT}MB, ram ${RTOT}MB; probe gpu ${GMB}MB rss ${RMB}MB per proc -> $NP procs" | tee "$OUT/sizing.txt"
PIDS=()
for p in $(seq 0 $((NP - 1))); do
  timeout "${CAP:-90m}" uv run --no-sync python scripts/turn_anchor.py "${A[@]}" --proc "$p" --nproc "$NP" \
    --out "$OUT/anchor/p$p.jsonl" > "$OUT/anchor-$p.txt" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || alert "anchor proc exit $?"; done
uv run --no-sync python - "$OUT" <<'PY' > "$OUT/summary.txt" 2>&1 || alert "anchor summary: $(tail -1 "$OUT/summary.txt")"
import glob, json, sys
from collections import defaultdict
rows = [json.loads(x) for f in glob.glob(f"{sys.argv[1]}/anchor/p*.jsonl") for x in open(f)]
if not rows:
    print("0 anchored records"); sys.exit(2)
b = defaultdict(lambda: [0, 0, 0.0])
for r in rows:
    b[r["bucket"]][0] += 1; b[r["bucket"]][1] += r["keep"]; b[r["bucket"]][2] += r["diff"]
bad = False
for k, (n, kept, d) in sorted(b.items()):
    print(f"bucket {k}: kept {kept}/{n} ({kept / n:.2f}), mean diff {d / n:+.3f}")
    bad |= kept / n < 0.3
print(f"total kept {sum(r['keep'] for r in rows)}/{len(rows)}")
sys.exit(2 if bad else 0)
PY
echo "$(date +%T) $(tr '\n' ' ' < "$OUT/summary.txt")" >> "$OUT/PROGRESS"
kill "$USAGE_PID" 2>/dev/null
echo done > "$OUT/DONE_ALL"
