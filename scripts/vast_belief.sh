#!/usr/bin/env bash
# Belief head (PLAN §0a "Hidden-information bound" (b)): train the variants on prebuilt belief shards (built on the
# laptop by scripts/build_belief_shards.py, pushed in data/cache/belief_shards) and report BCE vs the hypergeometric
# baseline. Env: INIT=<policy ckpt> (default data/cache/base_r3/best.pt), EPOCHS=3, PAR=<variants run at once>,
# VARIANTS="tag|args;tag|args" (default: h128, v1 window, no history, frozen-trunk h128).
# Usage: bash scripts/vast_belief.sh <name>. Do NOT run on the laptop (vast GPU only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
export HF_HOME="$PWD/data/cache/hf"
T0=$(date +%s)
uv sync --group dev --group ml -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
echo "deps install $(( $(date +%s) - T0 ))s" >> "$OUT/torch.txt"
{ nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; nproc; free -g | head -2;
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; } > "$OUT/machine.txt" 2>&1
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { echo "ALERT no CUDA torch" | tee -a "$OUT/ALERT"; exit 3; }
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3"G"}')"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
T0=$SECONDS
tick() { echo "$1 $((SECONDS - T0)) s" >> "$OUT/timing.txt"; }
INIT="${INIT:-data/cache/base_r3/best.pt}"
EPOCHS="${EPOCHS:-3}"
PAR="${PAR:-2}"
N_SHARDS=$(find data/cache/belief_shards -name '*.npz' | wc -l)
[ "$N_SHARDS" -gt 0 ] || { echo "ALERT no belief shards on the box" | tee -a "$OUT/ALERT"; exit 3; }
VARIANTS="${VARIANTS:-h128|--hist 128;v1|--hist 0;nohist|--hist 0 --nohist;frozen128|--hist 128 --freeze-trunk}"
run() {
  local TAG="$1" ARGS="$2"
  # shellcheck disable=SC2086
  uv run --no-sync python scripts/train_belief.py --shards data/cache/belief_shards --init "$INIT" --out "$OUT/$TAG" \
    --epochs "$EPOCHS" $ARGS > "$OUT/train-$TAG.txt" 2>&1 \
    || echo "ALERT train_belief $TAG failed: $(tail -1 "$OUT/train-$TAG.txt")" | tee -a "$OUT/ALERT"
  # each eval is appended to $OUT/PROGRESS by train_belief.py (the launcher relays ALERT/PROGRESS)
  tick "train-$TAG"
}
IFS=';' read -ra VS <<< "$VARIANTS"
i=0
for V in "${VS[@]}"; do
  run "${V%%|*}" "${V#*|}" &
  i=$((i + 1))
  if [ $((i % PAR)) -eq 0 ]; then wait; fi
done
wait
kill "$USAGE_PID" 2>/dev/null
echo done > "$OUT/DONE"
