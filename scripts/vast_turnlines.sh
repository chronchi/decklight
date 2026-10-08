#!/usr/bin/env bash
# Whole-turn lines on a vast box (user, 2026-10-07: "more lines on resource management/interactions with opponent"):
#   1. scripts/turn_lines.py: greedy + K sampled lines of one whole turn, opponent responding with the policy, from
#      (a) real-game DAGS starts (train decks on both sides; all turns) and (b) fresh turn-1 hands;
#   2. scripts/score_lines.py: every candidate played out through the opponent's next turn (S rollouts, board@h2).
# Env: CKPT, N_DAGS (2500), N_T1 (6 hands/deck), K (6), SAMPLES (8), GEN_CAP (50m), SCORE_CAP (100m), SEED (11).
# Usage: bash scripts/vast_turnlines.sh <name>. Results: runs/<name>/{dags,turn1}/lines.partial.jsonl, scored-*.jsonl
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_turnlines.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT"
: "${CKPT:?set CKPT}"; K=${K:-6}
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
GMB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RGB=$(free -g | awk '/Mem/{print $7}')
PR=$J; [ $(( GMB / 800 )) -lt $PR ] && PR=$(( GMB / 800 )); [ $(( RGB * 2 / 3 )) -lt $PR ] && PR=$(( RGB * 2 / 3 ))
echo "cores $J, gpu ${GMB}MB, ram avail ${RGB}G -> $PR procs" | tee "$OUT/sizing.txt"
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
# the loop must not hold the $(...) pipe open, or the caller waits forever (turnlines-20261007-1842 hung 40 min)
relay() { ( while sleep 300; do grep '^PROGRESS' "$1" 2>/dev/null | tail -1 | sed "s/^PROGRESS/PROGRESS $2/" >> "$OUT/PROGRESS"; done ) > /dev/null 2>&1 & echo $!; }
RC=0
for src in dags turn1; do
  if [ $src = dags ]; then ARGS=(--starts dags --n "${N_DAGS:-2500}"); else ARGS=(--starts turn1 --n "${N_T1:-6}"); fi
  RP=$(relay "$OUT/gen-$src.txt" "gen-$src")
  timeout "${GEN_CAP:-50m}" uv run --no-sync python scripts/turn_lines.py --ckpt "$CKPT" --decks train "${ARGS[@]}" \
    --k "$K" --seed "${SEED:-11}" --opp policy --out "$OUT/$src" --procs "$PR" --threads 1 --device cuda > "$OUT/gen-$src.txt" 2>&1
  rc=$?; kill $RP 2>/dev/null
  [ -f "$OUT/$src/ALERT" ] && sed "s/^/ALERT gen-$src /" "$OUT/$src/ALERT" >> "$OUT/ALERT"
  [ $rc -ne 0 ] && { alert "turn_lines $src exit $rc: $(tail -3 "$OUT/gen-$src.txt" | tr '\n' ' ')"; RC=$rc; }
  echo "PROGRESS gen-$src done rc=$rc rows=$(wc -l < "$OUT/$src/lines.partial.jsonl" 2>/dev/null) $(date +%T)" >> "$OUT/PROGRESS"
  [ -s "$OUT/$src/lines.partial.jsonl" ] || { alert "gen-$src: 0 rows, not scoring"; continue; }
  RP=$(relay "$OUT/score-$src.txt" "score-$src")
  timeout "${SCORE_CAP:-100m}" uv run --no-sync python scripts/score_lines.py --ckpt "$CKPT" \
    --lines "$OUT/$src/lines.partial.jsonl" --out "$OUT/scored-$src.jsonl" --cands $(( K + 1 )) \
    --samples "${SAMPLES:-8}" --horizon 2 --procs "$PR" --device cuda > "$OUT/score-$src.txt" 2>&1
  rc=$?; kill $RP 2>/dev/null
  [ -f "$OUT/scored-$src.jsonl.ALERT" ] && sed "s/^/ALERT score-$src /" "$OUT/scored-$src.jsonl.ALERT" >> "$OUT/ALERT"
  [ $rc -ne 0 ] && { alert "score_lines $src exit $rc: $(tail -3 "$OUT/score-$src.txt" | tr '\n' ' ')"; RC=$rc; }
  echo "PROGRESS score-$src done rc=$rc rows=$(wc -l < "$OUT/scored-$src.jsonl" 2>/dev/null) $(date +%T)" >> "$OUT/PROGRESS"
done
kill $USAGE_PID 2>/dev/null
exit $RC
