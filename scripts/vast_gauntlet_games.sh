#!/usr/bin/env bash
# Saved games on the gauntlet decks (BC vs ygo-agent + references) for the loss analysis (2026-10-05).
# Usage: bash scripts/vast_gauntlet_games.sh <name>. Checkpoint: data/cache/rl_init/best.pt (+ metrics.jsonl).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
J=$(nproc); [ "$J" -gt 16 ] && J=16
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  sleep 10
done
DEV=cpu; uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" && DEV=cuda
timeout 50m uv run --no-sync python scripts/gauntlet_games.py --ckpt data/cache/rl_init/best.pt --out "$OUT" \
  --device "$DEV" --threads "$J" > "$OUT/games.txt" 2>&1
echo "exit $?" >> "$OUT/games.txt"
echo done > "$OUT/DONE"
