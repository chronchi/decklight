#!/usr/bin/env bash
# M0 speed profiling (docs/plans/2026-10-16-speed-and-cheap-tests.md §1.3): where the time of a searched decision
# goes, on a GPU box. Product settings come from scripts/product.sh (P + hppo + value_p2 vs ygo-agent, deal seeds
# 9000+). Do NOT run on the laptop (vast GPU only).
#   phase A latency (LAT_MIN, default 8): 1 GPU proc --latency 1 --prof 1, plus a concurrent CPU-only run
#     (--device cpu) pinned to 4 other cores;
#   phase B throughput (B_MIN, default 8): production sizing via scripts/vast_search_match.sh with --prof 1, then
#     SIGTERM ends the procs (each dumps its in-flight prefixes to inflight-<proc>.jsonl and its partial counters
#     to prof-<proc>.json; the "exit 124" ALERT lines this causes are expected, noted in PROGRESS, not relayed);
#   phase C (C_MIN, default 6): like B with YGO_BLOCKING_SYNC=1 (blocking CUDA sync via cudaSetDeviceFlags) and
#     GPU_MARGIN=1.0.
# Env: LAT_MIN (8), B_MIN (8), C_MIN (6), GA_SEEDS (6), SMOKE=1 (2/3/2 min, GA_SEEDS=1).
# Usage: bash scripts/vast_speed_profile.sh <name>.
# Results: runs/<name>/{lat-gpu,lat-cpu,phase-b,phase-c}, PROGRESS, ALERT. ALERT lines: empty prof files and
# 0 searched decisions per phase (plus relayed failures).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export HF_HOME="$PWD/data/cache/hf"
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }
. scripts/product.sh
[ -s "$PRODUCT_CKPT" ] && [ -s "$PRODUCT_VALUE" ] || { alert "missing product ckpt or value"; exit 3; }
GS=${GA_SEEDS:-6}
[ "${SMOKE:-0}" = 1 ] && { LAT_MIN=2; B_MIN=3; C_MIN=2; GS=1; }
LAT_MIN=${LAT_MIN:-8}; B_MIN=${B_MIN:-8}; C_MIN=${C_MIN:-6}

# Phase A: latency, 1 GPU proc + a concurrent CPU-only run on 4 other cores (same deals, seeds 9000+).
LAT_ARGS="--ckpt $PRODUCT_CKPT --value $PRODUCT_VALUE --opp ygoagent --ga-seeds $GS --deal-seed0 9000 \
  $PRODUCT_SEARCH --prof 1 --latency 1 --nproc 1 --proc 0"
# shellcheck disable=SC2086
timeout "${LAT_MIN}m" uv run --no-sync python scripts/search_match.py $LAT_ARGS --device cuda \
  --out "$OUT/lat-gpu" > "$OUT/lat-gpu.txt" 2>&1 &
GPU_PID=$!
NC=$(nproc); CPUSET="$((NC - 4))-$((NC - 1))"
[ "$NC" -ge 5 ] || CPUSET="0-$((NC - 1))"  # tiny box: share everything rather than a negative range
# shellcheck disable=SC2086
taskset -c "$CPUSET" timeout "${LAT_MIN}m" uv run --no-sync python scripts/search_match.py $LAT_ARGS --device cpu \
  --out "$OUT/lat-cpu" > "$OUT/lat-cpu.txt" 2>&1 &
CPU_PID=$!
wait $GPU_PID; g=$?; wait $CPU_PID; c=$?
tick "phase A latency done (gpu exit $g, cpu exit $c on cores $CPUSET)"
echo "$(date +%T) phase A: width-1 runs cannot finish all deals in ${LAT_MIN}m, so exit 124 (timeout) is expected" \
  >> "$OUT/PROGRESS"
for d in lat-gpu lat-cpu; do
  if uv run --no-sync python scripts/search_match.py --summarize "$OUT/$d" > "$OUT/$d/summary.txt" 2>&1; then
    echo "$d $(head -1 "$OUT/$d/summary.txt")" >> "$OUT/PROGRESS"
  else
    alert "$d: summarize failed: $(tail -1 "$OUT/$d/summary.txt")"
  fi
  uv run --no-sync python scripts/audit/latency_report.py "$OUT/$d" > "$OUT/$d/latency.txt" 2>&1 \
    && sed "s/^/$(date +%T) $d /" "$OUT/$d/latency.txt" >> "$OUT/PROGRESS" \
    || alert "$d: $(tail -1 "$OUT/$d/latency.txt")"
  [ -s "$OUT/$d/prof-0.json" ] || alert "$d: empty or missing prof-0.json"
done

# Phase B: throughput at production sizing, --prof 1, SIGTERM at the cap (expected "exit 124" alerts stay in $tag/ALERT).
export VARIANTS="prof|--opp ygoagent --ga-seeds $GS $PRODUCT_SEARCH --prof 1"
VAR_CAP="${B_MIN}m" CKPT="$PRODUCT_CKPT" SM_ARGS="--value $PRODUCT_VALUE --deal-seed0 9000" \
  PROBE_NPROC=${PROBE_NPROC:-40} bash scripts/vast_search_match.sh "$NAME/phase-b"
echo "$(date +%T) phase B: timeout SIGTERM at ${B_MIN}m is expected; 'exit 124' lines in phase-b/ALERT are not problems" \
  >> "$OUT/PROGRESS"
tick "phase B throughput done"

# Phase C: like B with blocking CUDA sync and GPU_MARGIN=1.0.
YGO_BLOCKING_SYNC=1 GPU_MARGIN=1.0 VAR_CAP="${C_MIN}m" CKPT="$PRODUCT_CKPT" \
  SM_ARGS="--value $PRODUCT_VALUE --deal-seed0 9000" VARIANTS="prof-block|--opp ygoagent --ga-seeds $GS $PRODUCT_SEARCH --prof 1" \
  PROBE_NPROC=${PROBE_NPROC:-40} bash scripts/vast_search_match.sh "$NAME/phase-c"
echo "$(date +%T) phase C: YGO_BLOCKING_SYNC=1 GPU_MARGIN=1.0; timeout SIGTERM at ${C_MIN}m is expected" >> "$OUT/PROGRESS"
tick "phase C blocking-sync done"

# Per-phase ALERT lines: empty prof files, 0 searched decisions (SIGTERM exits filtered as expected).
for ph in phase-b phase-c; do
  arm=""
  for d in "$OUT/$ph"/*/; do  # the variant dir (prof / prof-block), whatever it is called
    if ls "$d"/games-*.jsonl >/dev/null 2>&1; then arm="$d"; break; fi
  done
  [ -n "$arm" ] || { alert "$ph: no variant dir with games"; continue; }
  nprof=$(ls "$arm"/prof-*.json 2>/dev/null | wc -l)
  [ "$nprof" -gt 0 ] || alert "$ph: no prof-*.json in $arm"
  nempty=$(find "$arm" -maxdepth 1 -name 'prof-*.json' -size 0 2>/dev/null | wc -l)
  [ "$nempty" -eq 0 ] || alert "$ph: $nempty empty prof files in $arm"
  nsearch=$(python3 -c "
import glob, json
n = 0
for f in glob.glob('$arm/games-*.jsonl'):
    for x in open(f):
        if x.strip():
            n += json.loads(x).get('n_searched', 0)
print(n)")
  [ "$nsearch" -gt 0 ] || alert "$ph: 0 searched decisions in $arm"
  echo "$(date +%T) $ph: $nsearch searched decisions, $nprof prof files" >> "$OUT/PROGRESS"
  if [ -f "$OUT/$ph/ALERT" ]; then  # relay real failures; the timeout's "exit 124" is the expected SIGTERM
    grep -v "proc pid .* exit 124" "$OUT/$ph/ALERT" >> "$OUT/ALERT" || true
  fi
  uv run --no-sync python scripts/audit/prof_report.py "$arm" > "$OUT/$ph/prof_report.txt" 2>&1 \
    && cat "$OUT/$ph/prof_report.txt" >> "$OUT/PROGRESS" \
    || alert "$ph: prof_report failed: $(tail -1 "$OUT/$ph/prof_report.txt")"
done
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
