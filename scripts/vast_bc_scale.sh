#!/usr/bin/env bash
# BC data-scaling run (from scratch, fixed recipe) on one vast GPU box:
#   1. CPU: expand every verified file that has no expansion yet (hand variants), 2. CPU: shards, 3. GPU: train.
# The card embeddings are shipped (data/cards/emb), not recomputed. Holdout = decks/pool/HOLDOUT + kewl-tune.
# Usage: bash scripts/vast_bc_scale.sh <name> [epochs]. Results in runs/<name>/ (pulled back by the launcher):
#   bc/metrics.jsonl, bc/best.pt, train.txt (first line = split sizes + decks), expanded/<dir>/ (new expansions).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; EPOCHS="${2:-15}"; OUT="runs/$NAME"; mkdir -p "$OUT/expanded"
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
if [ "$Q" = "max" ]; then J=$(nproc); else J=$(( Q / P )); fi
[ "$J" -lt 2 ] && J=2
uv sync --group dev --group ml -q
uv pip install -q --reinstall torch --index-url https://download.pytorch.org/whl/cu128 > "$OUT/torch.txt" 2>&1 || true
{ nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; echo "jobs $J";
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; } > "$OUT/machine.txt" 2>&1
T0=$SECONDS
# 1. expansion backlog: <src>/verified/<deck>.jsonl -> <src>/expanded-<deck>/ (skipped when any <src>/expanded*/<deck>[+tag].jsonl exists)
# EXPAND=0 (default since 2026-10-03): expansion runs continuously in the datagen driver; this job trains on what exists
[ "${EXPAND:-0}" = "1" ] && for v in data/demos/claude/verified/*.jsonl data/demos/planned*/verified/*.jsonl; do
  [ -s "$v" ] || continue
  src=$(dirname "$(dirname "$v")"); deck=$(basename "$v" .jsonl); dst="$src/expanded-$deck"
  uv run --no-sync python scripts/expand_lines.py --verified "$v" --lines-dir "$src/lines" --out "$dst" --variants 8 --jobs "$J" \
    --skip-done "$src/expanded*/*.jsonl" >> "$OUT/expand.txt" 2>&1
  if [ -d "$dst" ] && [ -n "$(ls -A "$dst")" ]; then cp -r "$dst" "$OUT/expanded/$(echo "$dst" | tr / _)"; fi
done
echo "expand $((SECONDS - T0)) s" > "$OUT/timing.txt"
# 2. shards from every record source (guides, claude, muse, games)
uv run --no-sync python scripts/build_bc_shards.py --out data/cache/bc_shards --jobs "$J" > "$OUT/shards.txt" 2>&1
echo "shards $((SECONDS - T0)) s" >> "$OUT/timing.txt"
# 3. BC from scratch, same recipe every run
HOLD="kewl-tune $( [ -f decks/pool/HOLDOUT ] && tr '\n' ' ' < decks/pool/HOLDOUT )"
uv run --no-sync python scripts/train_bc.py --shards data/cache/bc_shards --emb data/cards/emb/qwen3-0.6b-d512.npz \
  --out "$OUT/bc" --epochs "$EPOCHS" --batch 512 --lr 5e-4 --holdout $HOLD > "$OUT/train.txt" 2>&1
echo "train $((SECONDS - T0)) s" >> "$OUT/timing.txt"
rm -f "$OUT/bc/last.pt"
echo done > "$OUT/DONE"
