#!/usr/bin/env bash
# Turn-1 line headroom on a vast box: scripts/turn1_lines.py over the train, holdout and gauntlet splits in turn, all
# cores as worker processes. Env: CKPT, CRITIC, HANDS (20), K (32), MAX_TRAIN (40; 0 = all), T1_CAP (40m per split),
# SPLITS ("train holdout gauntlet"), SEED (0).
# Usage: bash scripts/vast_turn1.sh <name>. Results: runs/<name>/<split>/{lines.jsonl,summary.json}, <split>.txt, ALERT
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_turn1.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT"
: "${CKPT:?set CKPT}"; : "${CRITIC:?set CRITIC}"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
uv sync --group dev --group ml -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
J=$(nproc); [ "$Q" != max ] && J=$(( Q / P ))
# Workers: one CUDA context (~0.8 GB GPU) and ~1.5 GB RAM each (policy + critic + engine), capped by cores.
GMB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RGB=$(free -g | awk '/Mem/{print $7}')
PR=$J; [ $(( GMB / 800 )) -lt $PR ] && PR=$(( GMB / 800 )); [ $(( RGB * 2 / 3 )) -lt $PR ] && PR=$(( RGB * 2 / 3 ))
echo "cores $J, gpu ${GMB}MB, ram avail ${RGB}G -> $PR procs" > "$OUT/turn1.txt"
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
RC=0
for split in ${SPLITS:-train holdout gauntlet}; do
  MD=0; [ "$split" = train ] && MD="${MAX_TRAIN:-40}"
  # every 5 min the latest per-hand PROGRESS line goes to $OUT/PROGRESS (relayed into launcher.log by vast_job)
  ( while sleep 300; do tail -n 200 "$OUT/$split.txt" 2>/dev/null | grep '^PROGRESS' | tail -1 | sed "s/^PROGRESS/PROGRESS $split/" >> "$OUT/PROGRESS"; done ) &
  PROG_PID=$!
  timeout "${T1_CAP:-40m}" uv run --no-sync python scripts/turn1_lines.py --ckpt "$CKPT" --critic "$CRITIC" --decks "$split" \
    --hands "${HANDS:-20}" --k "${K:-32}" --seed "${SEED:-0}" --out "$OUT/$split" --procs "$PR" --threads 1 --device cuda \
    --max-decks "$MD" > "$OUT/$split.txt" 2>&1
  rc=$?; kill $PROG_PID 2>/dev/null
  # a capped split still leaves every finished hand in lines.partial.jsonl
  [ ! -f "$OUT/$split/lines.jsonl" ] && [ -f "$OUT/$split/lines.partial.jsonl" ] && alert "$split capped: $(wc -l < "$OUT/$split/lines.partial.jsonl") partial rows kept"
  [ $rc -ne 0 ] && { alert "turn1_lines $split exit $rc: $(tail -3 "$OUT/$split.txt" | tr '\n' ' ')"; RC=$rc; }
  [ -f "$OUT/$split/ALERT" ] && sed "s/^/ALERT $split: /" "$OUT/$split/ALERT" >> "$OUT/ALERT"
  echo "PROGRESS $split done rc=$rc $(date +%T)" | tee -a "$OUT/turn1.txt" >> "$OUT/PROGRESS"
done
kill $USAGE_PID 2>/dev/null
exit $RC
