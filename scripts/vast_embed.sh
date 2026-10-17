#!/usr/bin/env bash
# GPU box: embed card texts (MODE=strings: effect strings, ygo_zero.cards.effect_strings) with Qwen3-Embedding-0.6B. Output -> runs/$1/ (pulled back by the launcher).
# Incremental (custom cards): ONLY_MISSING=1 BASE=<base npz> OUT_NPZ=<merged npz path> YGO_EXTRA_CARDS=<extra dir>
# runs embed_cards.py --only-missing --base "$BASE" --out "$OUT_NPZ" (MODE=strings: the strings table).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
OUT="runs/$1"; mkdir -p "$OUT"
export HF_HOME="$PWD/data/cache/hf"
uv sync --group dev --group ml -q
# The ml group pins the CPU wheel (laptop); install a CUDA build over it here.
uv pip install -q --reinstall torch --index-url https://download.pytorch.org/whl/cu128 > "$OUT/torch.txt" 2>&1 || true
{ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; } > "$OUT/machine.txt" 2>&1
T0=$SECONDS
ARGS=(--dim 512 --batch 128)
if [ -n "${ONLY_MISSING:-}" ]; then
  ARGS+=(--only-missing --base "${BASE:?ONLY_MISSING needs BASE}" --out "${OUT_NPZ:?ONLY_MISSING needs OUT_NPZ}")
else
  ARGS+=(--out-dir "$OUT")
fi
if [ -n "${MODE:-}" ]; then
  ARGS+=(--"$MODE")
fi
uv run --no-sync python scripts/embed_cards.py "${ARGS[@]}" > "$OUT/embed.txt" 2>&1
echo "wall $((SECONDS - T0)) s" >> "$OUT/embed.txt"
echo done > "$OUT/DONE"
