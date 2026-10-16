#!/usr/bin/env bash
# Feedback games (user 2026-10-16: "replays for a few games in my laptop using P + search + value ... decks never seen,
# like 2 games, then also 2 games with decks seen ... P vs P using the best model we have now including the searches").
# The product (scripts/product.sh) on BOTH seats, CPU, one process per game, 1 thread each, cores 0-3.
# Unseen = decks/newcards (archetypes with 0 cards in any training data, ADR 0007), seen = training pool decks.
# Usage: bash scripts/launch/laptop-feedback-games.sh <name>
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
. scripts/product.sh
GAMES=("newcards goldpride:ragnaraika" "newcards dragunity:dinomorphia" "sample snake-eye:tearlaments" "sample sky-striker:dragon-link")
PIDS=()
for i in "${!GAMES[@]}"; do
  read -r SET PAIR <<< "${GAMES[$i]}"
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$i" uv run --no-sync python scripts/search_match.py \
    --ckpt "$PRODUCT_CKPT" --value "$PRODUCT_VALUE" --device cpu --opp self --both-search 1 $PRODUCT_SEARCH \
    --deck-set "$SET" --pairs "$PAIR" --seeds 1 --deal-seed0 "${DEAL_SEED0:-20261016}" --s-seat-only 0 --width 1 \
    --out "$OUT/g$i" > "$OUT/g$i.txt" 2>&1 &
  PIDS+=($!)
done
for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" || echo "ALERT game $i exit $?" | tee -a "$OUT/ALERT"; done
for i in "${!GAMES[@]}"; do
  [ -s "$OUT/g$i/games-0.jsonl" ] || echo "ALERT game $i (${GAMES[$i]}) wrote no record" | tee -a "$OUT/ALERT"
done
echo "$(date +%T) done: $(cat "$OUT"/g*/games-0.jsonl 2>/dev/null | wc -l) games" | tee -a "$OUT/PROGRESS"
