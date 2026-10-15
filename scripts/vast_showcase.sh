#!/usr/bin/env bash
# Showcase games (user 2026-10-15: "3 for each ... our network vs our network (the best product not just raw) ... one of
# the games with one of the decks being mitsurugi"): the product (P + hppo + value_p, guardfix settings) on BOTH seats
# (search_match --both-search 1), on PAIRS (pool decks, seat 0 first), deal seed 12345. One process per game.
# Env: PAIRS. Usage: bash scripts/vast_showcase.sh <name>. vast GPU only.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
PAIRS=${PAIRS:-mitsurugi-meta1:snake-eye,branded:tearlaments,blue-eyes:sky-striker}
IFS=',' read -ra PS <<< "$PAIRS"
PIDS=()
for i in "${!PS[@]}"; do
  YGO_FLIGHT_DIR="$OUT/g$i/flight" uv run --no-sync python scripts/search_match.py --ckpt data/cache/P/best.pt \
    --value data/cache/value_p/best.pt --device cuda --opp self --both-search 1 --honest 1 --k 4 --samples 4 \
    --horizon 1 --det-tries 32 --det-fallback skip --conf-skip 0.9 --pairs "${PS[$i]}" --seeds 1 --deal-seed0 12345 \
    --s-seat-only 0 --width 1 --out "$OUT/g$i" > "$OUT/g$i.txt" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || alert "game proc $pid exit $?"; done
for i in "${!PS[@]}"; do
  [ -s "$OUT/g$i/games-0.jsonl" ] || alert "game $i (${PS[$i]}) wrote no record: $(tail -1 "$OUT/g$i.txt")"
done
echo "$(date +%T) showcase done: $(cat "$OUT"/g*/games-0.jsonl 2>/dev/null | wc -l) games" | tee -a "$OUT/PROGRESS"
