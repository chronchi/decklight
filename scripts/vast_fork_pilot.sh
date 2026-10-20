#!/usr/bin/env bash
# M1 fork pilot (docs/plans/2026-10-20-rl-signal.md; pre-registered NOTEBOOK 2026-10-20): scripts/audit/fork_pilot.py
# over raw-P self-play games (runs/t1-boards-20261019-2309/raw, shipped as data/cache/fork-in/raw), 100 decisions per category
# (ht / place / t1 / any), played option + up to 3 alternatives on 16 shared honest worlds, scored by value_p2 at h1 and
# by the game result. Env: SAMPLES (16), N_PER (100), FCAP (45m per proc), MAX_ROLLOUT (800).
# Usage: bash scripts/vast_fork_pilot.sh <name>. Do NOT run on the laptop (vast GPU only).
# Results: runs/<name>/rows-*.jsonl, rows.pass1.json, rows.summary.json, summary.txt.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage}"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export HF_HOME="$PWD/data/cache/hf"
uv sync --group dev --group ml -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }
G=data/cache/fork-in/raw  # = runs/t1-boards-20261019-2309/raw, copied by the launcher
[ "$(ls $G/games-*.jsonl | wc -l)" -ge 1 ] || { alert "no games in $G"; exit 3; }
FP=(uv run --no-sync python scripts/audit/fork_pilot.py --games "$G" --ckpt data/cache/P/best.pt
    --value data/cache/value_p2/best.pt --samples "${SAMPLES:-16}" --n-per "${N_PER:-100}"
    --max-rollout "${MAX_ROLLOUT:-800}" --out "$OUT/rows.jsonl")
# probe: one proc (after its pass 1) for 60 s measures GPU memory and RSS per process (its rows are discarded)
"${FP[@]}" --nproc 40 --proc 0 --out "$OUT/probe/rows.jsonl" > "$OUT/probe.txt" 2>&1 &
PP=$!; GMB=0; RMB=0
for _ in $(seq 200); do  # pass 1 (engine replay, ~1-2 min) first, then 60 s of forks
  sleep 3
  kill -0 $PP 2>/dev/null || break
  grep -q "decisions/game" "$OUT/probe.txt" && { [ -z "${PT:-}" ] && PT=$SECONDS; [ $((SECONDS - PT)) -ge 60 ] && break; }
  g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  r=$(ps -o rss= --ppid $PP 2>/dev/null | sort -n | tail -1)
  [ "${g:-0}" -gt "$GMB" ] && GMB=$g; [ $(( ${r:-0} / 1024 )) -gt "$RMB" ] && RMB=$(( ${r:-0} / 1024 ))
done
kill $PP 2>/dev/null; wait $PP 2>/dev/null
grep -q "decisions/game" "$OUT/probe.txt" || { alert "probe: pass 1 did not run: $(tail -1 "$OUT/probe.txt")"; exit 3; }
[ "$GMB" -gt 0 ] || { alert "probe: no GPU memory reading (see probe.txt)"; exit 3; }
. scripts/cores.sh
J=$(box_cores)
GTOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RTOT=$(free -m | awk '/Mem/{print $2}')
NP=$(python3 -c "print(max(1, min($J, int($GTOT*0.85/($GMB*1.3)), int($RTOT*0.8/($RMB*1.3)))))")
echo "cores $J, gpu ${GTOT}MB, ram ${RTOT}MB; probe gpu ${GMB}MB rss ${RMB}MB per proc -> $NP procs" | tee "$OUT/sizing.txt"
tick "sizing: $NP procs"
PIDS=()
for p in $(seq 0 $((NP - 1))); do
  timeout "${FCAP:-45m}" "${FP[@]}" --nproc "$NP" --proc "$p" > "$OUT/proc-$p.txt" 2>&1 &
  PIDS+=($!)
done
# progress: rows written every 2 min; at the first ~10% a partial summary checks the numbers move sensibly
CHECKED=0
while :; do
  alive=0; for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
  n=$(cat "$OUT"/rows-*.jsonl 2>/dev/null | grep -c .)
  tick "rows $n"
  if [ "$CHECKED" = 0 ] && [ "$n" -ge 40 ]; then
    "${FP[@]}" --summarize > "$OUT/summary-partial.txt" 2>&1 || alert "partial summary failed: $(tail -1 "$OUT/summary-partial.txt")"
    grep -h "ALERT" "$OUT/summary-partial.txt" >> "$OUT/ALERT"
    echo "PROGRESS partial $(head -1 "$OUT/summary-partial.txt")" >> "$OUT/PROGRESS"
    CHECKED=1
  fi
  [ "$alive" = 0 ] && break
  sleep 120
done
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"; rc=$?
  [ $rc -ne 0 ] && alert "fork proc $i exit $rc: $(tail -1 "$OUT/proc-$i.txt")"
done
tick "forks done"
[ -s "$OUT/rows.pass1.json" ] || alert "no rows.pass1.json (proc 0 writes it after pass 1)"
"${FP[@]}" --summarize > "$OUT/summary.txt" 2>&1 || alert "summary failed: $(tail -1 "$OUT/summary.txt")"
grep -h "ALERT" "$OUT/summary.txt" >> "$OUT/ALERT"
cat "$OUT/summary.txt" | sed 's/^/PROGRESS /' >> "$OUT/PROGRESS"
tick done
