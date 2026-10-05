#!/usr/bin/env bash
# Turn-1 search teacher (ADR 0006 M2) on a many-core GPU box. Usage: bash scripts/vast_search.sh <name>
# 1. combo check: BC policy greedy vs searched (top-K rollouts) vs expert on the playtest combo hands (CPU procs)
# 2. generate HANDS searched turn-1 records over PROCS processes (CPU, 1 torch thread each)
# 3. distill: vast_bc_gold.sh with VARIANTS="zero ft-search" (BASE = $BASE), report cards for both
# Env: BASE (default data/cache/rl_init/best.pt), HANDS (default 3000), K (default 4), PROCS (default cores - 2),
# DISTILL=0 stops after step 2. Results: runs/<name>/{combo-*.json,search.txt,search/p<k>/,...}.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT/search"
BASE="${BASE:-data/cache/rl_init/best.pt}"; K="${K:-4}"; HANDS="${HANDS:-3000}"
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
N=$(nproc); P=${PROCS:-$(( N - 2 ))}; [ "$P" -lt 1 ] && P=1; PER=$(( (HANDS + P - 1) / P ))
echo "procs $P hands/proc $PER cores $N k $K base $BASE" > "$OUT/search.txt"
T0=$SECONDS
# 1. combo check, 8 decks split over 4 procs by seed-sampled decks: one proc per (seed) sample of 8 decks x 3 hands
for s in 0 1 2 3; do
  timeout 40m uv run --no-sync python scripts/gen_search.py --ckpt "$BASE" --k "$K" --combo 8 --hands 3 --seed "$s" \
    --out "$OUT/combo-s$s.json" > "$OUT/combo-s$s.txt" 2>&1 &
done
# 2. generation (runs alongside the combo check)
for p in $(seq 0 $((P - 5))); do
  timeout 90m uv run --no-sync python scripts/gen_search.py --ckpt "$BASE" --k "$K" --hands "$PER" \
    --seed $(( 1000000 + p * 10000 )) --out "$OUT/search/p$p" > "$OUT/search-$p.txt" 2>&1 &
done
wait
echo "search wall $((SECONDS - T0)) s" >> "$OUT/search.txt"
grep -h "combo summary" "$OUT"/combo-s*.txt >> "$OUT/search.txt"
grep -h "^hands \|Error\|Traceback" "$OUT"/search-*.txt >> "$OUT/search.txt"
find "$OUT/search" -name "*.jsonl" -exec cat {} + | wc -l >> "$OUT/search.txt"
[ "${DISTILL:-1}" = "1" ] || { echo done > "$OUT/DONE"; exit 0; }
mkdir -p data/demos/search
for d in "$OUT"/search/p*; do p=$(basename "$d"); for f in "$d"/*.jsonl; do [ -f "$f" ] && cp "$f" "data/demos/search/$p-$(basename "$f")"; done; done
BASE="$BASE" VARIANTS="zero ft-search" PLAY_ALL=1 SKIP_SCRATCH=1 PT_ZERO=0 \
  PT_ARGS="--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent" bash scripts/vast_bc_gold.sh "$NAME"
