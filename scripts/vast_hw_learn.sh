#!/usr/bin/env bash
# Decisive-label learnability test (NOTEBOOK 2026-10-20 "hw learnability, pre-registered"): fine-tune P on the 1,483
# Stage L decisive labels only (data/demos/sw-hw, kinds sw-hw / sw-hw_own, replication 0.895), KL-anchored to P, then
# compare P and each student label by label on held-out labels: val = unseen games of trained decks, holdout = every
# 5th hw deck (sorted), never trained on. No games are played. Env: ARMS ("kl03:0.3 kl1:1.0"), EPOCHS (10), BASE.
# Usage: bash scripts/vast_hw_learn.sh <name>. Do NOT run on the laptop (vast GPU only).
# Results: runs/<name>/<arm>/{metrics.jsonl,final.pt}, runs/<name>/flips-<arm>.{txt,json}, hw_holdout.txt.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export HF_HOME="$PWD/data/cache/hf"
uv sync --group dev --group ml -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
T0=$SECONDS
tick() { echo "$1 $((SECONDS - T0)) s" >> "$OUT/timing.txt"; }
BASE="${BASE:-data/cache/P/best.pt}"
SH=data/cache/hw_shards
uv run --no-sync python scripts/build_bc_shards.py --glob 'data/demos/sw-hw/*.jsonl' --strict-labels --out "$SH" \
  --jobs "$(nproc)" > "$OUT/shards.txt" 2>&1
tick shards
grep -q "^samples: 1483$" "$OUT/shards.txt" || { alert "hw shards: expected 1483 samples, got: $(tail -1 "$OUT/shards.txt")"; exit 3; }
uv run --no-sync python -c "
from pathlib import Path
from ygo_zero.model.bc_data import load_shards
d = load_shards(Path('$SH'))
print(' '.join(sorted(set(map(str, d['deck'])))[::5]))" > "$OUT/hw_holdout.txt"
[ "$(wc -w < "$OUT/hw_holdout.txt")" -ge 20 ] || { alert "hw holdout deck list too short"; exit 3; }
# the official holdout decks are not in sw-hw (pool train decks only); passing them too keeps train_bc's check honest
HOLD="$(cat "$OUT/hw_holdout.txt") kewl-tune $( [ -f decks/pool/HOLDOUT ] && tr '\n' ' ' < decks/pool/HOLDOUT )"
VF=(--val-frac-kind sw-hw=0.25 sw-hw_own=0.25)
for A in ${ARMS:-kl03:0.3 kl1:1.0}; do
  ARM=${A%%:*}; KL=${A#*:}
  # shellcheck disable=SC2086
  uv run --no-sync python scripts/train_bc.py --shards "$SH" --emb data/cards/emb/qwen3-0.6b-d512.npz --holdout $HOLD \
    "${VF[@]}" --init "$BASE" --epochs "${EPOCHS:-10}" --lr 1e-4 --batch 64 --select-on last \
    --kinds sw-hw sw-hw_own --require-kinds sw-hw sw-hw_own --kl-init "$KL" --out "$OUT/$ARM" > "$OUT/train-$ARM.txt" 2>&1
  rc=$?
  rm -f "$OUT/$ARM/last.pt"
  tick "train-$ARM"
  grep -h "^ALERT" "$OUT/train-$ARM.txt" >> "$OUT/ALERT"
  [ -s "$OUT/$ARM/final.pt" ] || { alert "train-$ARM wrote no final.pt (exit $rc): $(tail -1 "$OUT/train-$ARM.txt")"; continue; }
  # shellcheck disable=SC2086
  uv run --no-sync python scripts/eval_flips.py --shards "$SH" --emb data/cards/emb/qwen3-0.6b-d512.npz --base "$BASE" \
    --student "$OUT/$ARM/final.pt" --kinds sw-hw sw-hw_own --holdout $HOLD "${VF[@]}" --out "$OUT/flips-$ARM.json" \
    > "$OUT/flips-$ARM.txt" 2>&1 || alert "eval_flips $ARM exit $?: $(tail -1 "$OUT/flips-$ARM.txt")"
  echo "$ARM $(grep -E '^(val|holdout) +all' "$OUT/flips-$ARM.txt" | tr -s ' ' | tr '\n' ';')" >> "$OUT/PROGRESS"
  tick "flips-$ARM"
done
echo done >> "$OUT/timing.txt"
