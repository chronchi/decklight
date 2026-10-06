#!/usr/bin/env bash
# GPU throughput benchmark for every RL stage (scripts/bench_rl.py) on one GPU box.
# Env: BASE (BC checkpoint best.pt, required), SECTIONS, BENCH_S, RL_ITERS, RL_ARGS (extra train_rl flags).
# Usage: bash scripts/vast_bench_rl.sh <name>. Results: runs/<name>/bench.jsonl, bench.txt, ALERT, DONE.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_bench_rl.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT"
: "${BASE:?set BASE to the BC checkpoint best.pt}"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
[ -f "$BASE" ] || { alert "missing $BASE"; exit 3; }
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg)"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
{ echo "cpus $(nproc)"; echo "--- free -g ---"; free -g; echo "--- gpus ---"; nvidia-smi --query-gpu=name,memory.total --format=csv; } > "$OUT/machine.txt" 2>&1
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
TH=$(nproc); [ "$Q" != max ] && TH=$(( Q / P )); echo "threads $TH" >> "$OUT/machine.txt"
SECTIONS=${SECTIONS:-rollout update}
timeout 30m uv run --no-sync python scripts/bench_rl.py --ckpt "$BASE" --out "$OUT/bench.jsonl" --threads "$TH" \
  --seconds "${BENCH_S:-30}" --sections $SECTIONS > "$OUT/bench.txt" 2>&1
grep -E "^ALERT|Traceback" "$OUT/bench.txt" >> "$OUT/ALERT" || true
# end-to-end: a few real PPO iterations (rollout + update + final eval vs BC), as train_rl runs them
timeout 30m uv run --no-sync python scripts/train_rl.py --init "$BASE" --out "$OUT/rl" --threads "$TH" \
  --iters "${RL_ITERS:-4}" --value-warmup 1 --eval-every 1000 --no-ygoagent ${RL_ARGS:-} > "$OUT/rl.txt" 2>&1 \
  || echo "ALERT train_rl exit $?" >> "$OUT/ALERT"
grep -E "^ALERT|Traceback" "$OUT/rl.txt" >> "$OUT/ALERT" || true
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
