#!/usr/bin/env bash
# Mid-game teacher data (ADR 0006 M3 + DAgger) on a many-core GPU box, generation only (distill is a separate job).
# Usage: bash scripts/vast_teachers.sh <name>
# 1. DAgger: our policy (BASE) plays ygo-agent on decks/gauntlet, ygo-agent labels every student decision
#    (scripts/gen_dagger.py; DAGGER_GAMES over DPROCS processes, beta 0.3)
# 2. window search on recorded games: ygo-agent teacher games + gold games + the new DAgger games
#    (scripts/gen_window.py; WINDOW_GAMES sampled, WPROCS GPU processes x width 32, k 4, horizon 1, own-frac 0.1)
# Env: BASE (default data/cache/rl_init/best.pt), DAGGER_GAMES (2000), DPROCS (16), WINDOW_GAMES (3000), WPROCS (8).
# Results: runs/<name>/{dagger/p<k>/, window/p<k>/, teachers.txt, *.txt}
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
BASE="${BASE:-data/cache/rl_init/best.pt}"
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
DG=${DAGGER_GAMES:-2000}; DP=${DPROCS:-16}; DPER=$(( (DG + DP - 1) / DP ))
WG=${WINDOW_GAMES:-3000}; WP=${WPROCS:-8}
echo "dagger $DG games / $DP procs; window $WG games / $WP procs; base $BASE; cores $(nproc)" > "$OUT/teachers.txt"
T0=$SECONDS
for p in $(seq 0 $((DP - 1))); do
  timeout 40m uv run --no-sync python scripts/gen_dagger.py --ckpt "$BASE" --device cuda --games "$DPER" --beta 0.3 \
    --seed $(( 2000000 + p * 10000 )) --threads 2 --out "$OUT/dagger/p$p" > "$OUT/dagger-$p.txt" 2>&1 &
done
wait
echo "dagger wall $((SECONDS - T0)) s" >> "$OUT/teachers.txt"
grep -h "games\|Traceback\|Error" "$OUT"/dagger-*.txt | tail -n 40 >> "$OUT/teachers.txt"
mkdir -p data/demos/dagger
for d in "$OUT"/dagger/p*; do p=$(basename "$d"); for f in "$d"/*.jsonl; do [ -f "$f" ] && cp "$f" "data/demos/dagger/$p-$(basename "$f")"; done; done
T1=$SECONDS
for p in $(seq 0 $((WP - 1))); do
  timeout 70m uv run --no-sync python scripts/gen_window.py --ckpt "$BASE" --device cuda --width 32 \
    --records 'data/demos/teacher/*.jsonl' 'data/demos/gold/*.jsonl' 'data/demos/dagger/*.jsonl' --games "$WG" --proc "$p" --nproc "$WP" \
    --k 4 --horizon 1 --own-frac 0.1 --seed 5 --out "$OUT/window/p$p" > "$OUT/window-$p.txt" 2>&1 &
done
wait
echo "window wall $((SECONDS - T1)) s" >> "$OUT/teachers.txt"
grep -h "^done\|Traceback\|Error" "$OUT"/window-*.txt >> "$OUT/teachers.txt"
du -sh "$OUT/dagger" "$OUT/window" >> "$OUT/teachers.txt"
echo done > "$OUT/DONE"
