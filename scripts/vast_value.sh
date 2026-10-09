#!/usr/bin/env bash
# Learned value v2 (ADR 0007, Labellers v2): outcome shards + value training + calibration.
# Replaces the hand-written position_score in search with P(acting player wins | state).
# Env: INIT=<policy ckpt> (default data/cache/bc_base/best.pt), RECORDS=<record globs>
# (default "data/demos/teacher/*.jsonl"), STRIDE=4, EPOCHS=4, EMB=<embeddings npz>.
# Usage: bash scripts/vast_value.sh <name>. Do NOT run on the laptop (vast GPU only).
# Results in runs/<name>/value/ (pulled back by the launcher): best.pt + metrics.jsonl.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
if [ "$Q" = "max" ]; then J=$(nproc); else J=$(( Q / P )); fi
[ "$J" -lt 2 ] && J=2
export HF_HOME="$PWD/data/cache/hf"
# CUDA torch straight away (the lock pins the CPU wheel: skip it instead of downloading both; 2026-10-05)
T0=$(date +%s)
uv sync --group dev --group ml -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
echo "deps install $(( $(date +%s) - T0 ))s" >> "$OUT/torch.txt"
{ nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; echo "jobs $J";
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; } > "$OUT/machine.txt" 2>&1
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA torch: abort" | tee -a "$OUT/torch.txt"; exit 3; }
# GPU/CPU usage every 30 s (is a better GPU worth it? 2026-10-05)
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg)"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
T0=$SECONDS
tick() { echo "$1 $((SECONDS - T0)) s" >> "$OUT/timing.txt"; }
INIT="${INIT:-data/cache/bc_base/best.pt}"
RECORDS="${RECORDS:-data/demos/teacher/*.jsonl}"
STRIDE="${STRIDE:-4}"
EPOCHS="${EPOCHS:-4}"
EMB="${EMB:-data/cards/emb/qwen3-0.6b-d512.npz}"
HOLD="kewl-tune $( [ -f decks/pool/HOLDOUT ] && tr '\n' ' ' < decks/pool/HOLDOUT )"
# shellcheck disable=SC2086
uv run --no-sync python scripts/build_value_shards.py --records $RECORDS \
  --out data/cache/value_shards --stride "$STRIDE" --jobs "$J" > "$OUT/shards.txt" 2>&1
tick shards
# VARIANTS="tag:init:lr:epochs ..." trains several values on the same shards into $OUT/<tag>/ (default: one variant
# "value" = $INIT, lr 1e-4, $EPOCHS: the old single-run layout). An init with a value head is scored first (epoch "init").
VARIANTS="${VARIANTS:-value:$INIT:1e-4:$EPOCHS}"
for V in $VARIANTS; do
IFS=: read -r TAG VINIT VLR VEP <<< "$V"
[ -s "$VINIT" ] || { echo "ALERT missing init $VINIT" | tee -a "$OUT/ALERT"; continue; }
uv run --no-sync python scripts/train_value.py --shards data/cache/value_shards --emb "$EMB" \
  --init "$VINIT" --out "$OUT/$TAG" --epochs "$VEP" --batch 256 --lr "$VLR" \
  --holdout $HOLD > "$OUT/train-$TAG.txt" 2>&1 || echo "ALERT train_value $TAG failed: $(tail -1 "$OUT/train-$TAG.txt")" | tee -a "$OUT/ALERT"
tick "train-$TAG"
rm -f "$OUT/$TAG/last.pt"
[ -s "$OUT/$TAG/metrics.jsonl" ] || continue
# final calibration table (also printed at the end of train-<tag>.txt)
uv run --no-sync python - "$OUT/$TAG/metrics.jsonl" >> "$OUT/train-$TAG.txt" <<'PY'
import json, sys
recs = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
recs = [r for r in recs if "val" in r or "train" in r]
last = recs[-1]
for split in ("train", "val", "holdout"):
    if split not in last:
        continue
    m = last[split]
    print(f"calibration {split} (n={m['n']} mse={m['mse']:.4f} acc={m['acc']:.4f} "
          f"ece={m['cal']['ece']:.4f} hacc={m['hacc']:.4f}):")
    for i, (p, w, c) in enumerate(m["cal"]["bins"]):
        ps = f"{p:.3f}" if p is not None else "  - "
        ws = f"{w:.3f}" if w is not None else "  - "
        print(f"  p{i / 10:.1f}-{(i + 1) / 10:.1f}: mean_p {ps} win_share {ws} n {c}")
PY
done
tick calibration
kill "$USAGE_PID" 2>/dev/null
echo done > "$OUT/DONE"
