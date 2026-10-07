#!/usr/bin/env bash
# Score turn-1 lines by play-out on a vast box: scripts/score_lines.py with one worker per ~0.8 GB of GPU memory.
# Env: CKPT, LINES (lines.jsonl), CANDS (4), SAMPLES (8), HORIZON (2), CAP (90m).
# Usage: bash scripts/vast_score.sh <name>. Results: runs/<name>/scored.jsonl(+.summary.json), score.txt, ALERT, PROGRESS
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_score.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT"
: "${CKPT:?set CKPT}"; : "${LINES:?set LINES}"
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
echo "cores $J, gpu ${GMB}MB, ram avail ${RGB}G -> $PR procs" > "$OUT/score.txt"
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
( while sleep 300; do grep '^PROGRESS' "$OUT/score.log" 2>/dev/null | tail -1 >> "$OUT/PROGRESS"; done ) &
PROG_PID=$!
timeout "${CAP:-90m}" uv run --no-sync python scripts/score_lines.py --ckpt "$CKPT" --lines "$LINES" --out "$OUT/scored.jsonl" \
  --cands "${CANDS:-4}" --samples "${SAMPLES:-8}" --horizon "${HORIZON:-2}" --procs "$PR" --device cuda > "$OUT/score.log" 2>&1
rc=$?; kill $PROG_PID $USAGE_PID 2>/dev/null
[ -f "$OUT/scored.jsonl.ALERT" ] && sed 's/^/ALERT /' "$OUT/scored.jsonl.ALERT" >> "$OUT/ALERT"
[ $rc -ne 0 ] && alert "score_lines exit $rc: $(tail -3 "$OUT/score.log" | tr '\n' ' ')"
echo "PROGRESS score done rc=$rc $(date +%T)" >> "$OUT/PROGRESS"
exit $rc
