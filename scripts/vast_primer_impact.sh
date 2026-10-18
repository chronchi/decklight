#!/usr/bin/env bash
# Impact scoring (primer-impact): build_primers.py --score over the hppo + primer game dirs (GPU value net).
# Sizing: procs = min(cores, RAM*0.8 / per-proc RSS, GPU memory*0.85 / per-proc GPU mem) with per-proc numbers
# measured by a <=60 s probe (one --score run on 60 games: peak nvidia-smi used_memory + RSS), like
# vast_score_decisions.sh. A scored game with bad w (NaN/outside [0,1]) fails the job (build_primers exits 2).
# Usage: bash scripts/vast_primer_impact.sh <name> <game dirs...>
# Results: runs/<name>/summaries.json.gz and runs/<name>/primers.json, PROGRESS, ALERT.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_primer_impact.sh <name> <game dirs...>}"; shift; OUT="runs/$NAME"; mkdir -p "$OUT"
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
. scripts/cores.sh
J=$(box_cores); RTOT=$(free -m | awk '/Mem/{print $2}')
# Probe: one proc scores 60 games (<=60 s of sampling); its peak GPU memory and RSS size the pool.
FIRST="$1"
uv run --no-sync python scripts/build_primers.py --score --device cuda --procs 1 --max-games 60 --min-games 1 \
  --games "$FIRST" --summaries "$OUT/probe-summ.json.gz" --out "$OUT/probe-prim.json" > "$OUT/probe.txt" 2>&1 &
PP=$!; GMB=0; RMB=0
for _ in $(seq 20); do
  sleep 3
  kill -0 $PP 2>/dev/null || break
  g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  r=$(ps -o rss= --ppid $PP 2>/dev/null | sort -n | tail -1)
  [ "${g:-0}" -gt "$GMB" ] && GMB=$g; [ $(( ${r:-0} / 1024 )) -gt "$RMB" ] && RMB=$(( ${r:-0} / 1024 ))
done
wait $PP || { alert "probe failed: $(tail -1 "$OUT/probe.txt")"; exit 3; }
GTOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$GMB" -gt 0 ] || { alert "probe: no GPU memory reading (see probe.txt)"; exit 3; }
[ "$RMB" -gt 0 ] || RMB=2000
NP=$(python3 -c "print(max(1, min($J, int($GTOT*0.85/($GMB*1.5)), int($RTOT*0.8/($RMB*1.3)))))")
echo "cores $J, gpu ${GTOT}MB, ram ${RTOT}MB; probe gpu ${GMB}MB rss ${RMB}MB per proc -> $NP procs" | tee "$OUT/sizing.txt"
echo "$(date +%T) scoring $* on $NP procs" >> "$OUT/PROGRESS"
# PROGRESS / ALERT / impact lines stream live (the launcher relays them within one poll), not after the run.
uv run --no-sync python scripts/build_primers.py --score --device cuda --procs "$NP" --min-games 1 \
  --games "$@" --summaries "$OUT/summaries.json.gz" --out "$OUT/primers.json" 2>&1 | tee "$OUT/score.log" | \
  while IFS= read -r line; do
    case "$line" in
      ALERT*) echo "$line" >> "$OUT/ALERT" ;;
      PROGRESS*|impact:*) echo "$(date +%T) $line" >> "$OUT/PROGRESS" ;;
    esac
  done
rc=${PIPESTATUS[0]}
grep -l "Traceback" "$OUT/score.log" 2>/dev/null | head -1 | while read -r l; do alert "traceback in $l: $(grep -m1 -A0 'Error' "$l")"; done
[ $rc -eq 0 ] || { alert "build_primers --score exited $rc (see score.log)"; exit 3; }
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
