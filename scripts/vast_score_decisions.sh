#!/usr/bin/env bash
# N2a scoring (engine plan, milestone N2a): the product value net on every decision of both
# seats of the value-p2 card arms (P + hppo with value_p2 / value_p vs ygo-agent, seeds 9000+),
# then annotate_games.py --values for both arms.
#   inputs (shipped by the launcher via --data): data/cache/score-in/{hppo-p2,hppo-p}/games-*.jsonl
#     (22 files each), data/cache/value_p2/best.pt (the product value), data/cards/emb
#   outputs in runs/<name>/: values-<arm>.jsonl (merged shards) + ann-<arm>/ (the launcher pulls
#     runs/<name>/ back).
# Env: SCORE_CAP (40m per scoring arm). Usage: bash scripts/vast_score_decisions.sh <name>.
# Do NOT run on the laptop (vast GPU only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_score_decisions.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export HF_HOME="$PWD/data/cache/hf"
uv sync --group dev --group ml -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }
IN=data/cache/score-in; VAL=data/cache/value_p2/best.pt
[ -s "$VAL" ] || { alert "missing product value $VAL"; exit 3; }
for arm in hppo-p2 hppo-p; do
  ls "$IN/$arm"/games-*.jsonl > /dev/null 2>&1 || { alert "no games in $IN/$arm"; exit 3; }
done
. scripts/cores.sh
J=$(box_cores)
# probe: one process scores 8 games; its peak GPU memory and RSS size the pool
uv run --no-sync python scripts/score_decisions.py --games "$IN/hppo-p2" --value "$VAL" \
  --out "$OUT/probe.jsonl" --device cuda --max-games 8 --batch 2048 > "$OUT/probe.txt" 2>&1 &
PP=$!; GMB=0; RMB=0
for _ in $(seq 12); do
  sleep 10
  kill -0 $PP 2>/dev/null || break
  g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  r=$(ps -o rss= --ppid $PP 2>/dev/null | sort -n | tail -1)
  [ "${g:-0}" -gt "$GMB" ] && GMB=$g; [ $(( ${r:-0} / 1024 )) -gt "$RMB" ] && RMB=$(( ${r:-0} / 1024 ))
done
wait $PP || { alert "probe failed: $(tail -1 "$OUT/probe.txt")"; exit 3; }
GTOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RTOT=$(free -m | awk '/Mem/{print $2}')
[ "$GMB" -gt 0 ] || { alert "probe: no GPU memory reading (see probe.txt)"; exit 3; }
[ "$RMB" -gt 0 ] || RMB=2000
NP=$(python3 -c "print(max(1, min($J, int($GTOT*0.85/($GMB*1.5)), int($RTOT*0.8/($RMB*1.3)))))")
echo "cores $J, gpu ${GTOT}MB, ram ${RTOT}MB; probe gpu ${GMB}MB rss ${RMB}MB per proc -> $NP procs" | tee "$OUT/sizing.txt"
tick "sizing: $NP procs"
for arm in hppo-p2 hppo-p; do
  d="$OUT/$arm"; mkdir -p "$d"
  PIDS=()
  for ((p = 0; p < NP; p++)); do
    timeout "${SCORE_CAP:-40m}" uv run --no-sync python scripts/score_decisions.py --games "$IN/$arm" \
      --value "$VAL" --out "$d/values-$p.jsonl" --device cuda --batch 2048 --shard "$p/$NP" \
      > "$d/score-$p.txt" 2>&1 &
    PIDS+=($!)
  done
  # first milestone: the first shard file with >= 50 games -> decisions/s to PROGRESS
  MS=0
  while kill -0 "${PIDS[@]}" 2>/dev/null; do
    sleep 15
    for f in "$d"/values-*.jsonl; do
      [ -f "$f" ] || continue
      n=$(wc -l < "$f")
      if [ "$MS" = 0 ] && [ "$n" -ge 50 ]; then
        MS=1
        stat=$(python3 -c "
import json
n = d = 0
for x in open('$f'):
    r = json.loads(x); n += 1; d += len(r['k'])
print(f'{n} {d}')")
        echo "$(date +%T) milestone $arm first shard: $stat (games decisions) in $((SECONDS - T0)) s" >> "$OUT/PROGRESS"
        break
      fi
    done
    [ "$MS" = 1 ] && break
  done
  rc=0
  for pid in "${PIDS[@]}"; do wait "$pid" || { alert "$arm: score proc pid $pid exit $?"; rc=1; }; done
  [ $rc -eq 0 ] || exit 4
  grep -h "^ALERT\|Traceback" "$d"/score-*.txt >> "$OUT/ALERT" 2>/dev/null || true
  want=$(cat "$IN/$arm"/games-*.jsonl | wc -l)
  cat "$d"/values-*.jsonl > "$OUT/values-$arm.jsonl"
  got=$(wc -l < "$OUT/values-$arm.jsonl")
  [ "$got" -eq "$want" ] || { alert "$arm: scored $got games, input has $want"; exit 4; }
  if [ "$MS" = 0 ]; then  # shards smaller than 50 games each: milestone from the merged file
    stat=$(python3 -c "
import json
n = d = 0
for x in open('$OUT/values-$arm.jsonl'):
    r = json.loads(x); n += 1; d += len(r['k'])
print(f'{n} {d}')")
    echo "$(date +%T) milestone $arm merged: $stat (games decisions) in $((SECONDS - T0)) s" >> "$OUT/PROGRESS"
  fi
  tick "$arm scored $got games"
  uv run --no-sync python scripts/annotate_games.py --games "$IN/$arm" --values "$OUT/values-$arm.jsonl" \
    --out "$OUT/ann-$arm" --workers "$J" > "$d/annotate.txt" 2>&1 || { alert "$arm: annotate failed: $(tail -1 "$d/annotate.txt")"; exit 4; }
  tail -4 "$d/annotate.txt" | sed "s/^/$(date +%T) $arm /" >> "$OUT/PROGRESS"
done
echo done > "$OUT/DONE"
