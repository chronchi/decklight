#!/usr/bin/env bash
# League self-play states + outcomes (ADR 0007) on a many-core GPU box, generation only.
# Usage: bash scripts/vast_selfplay.sh <name>
# Student (CKPT, sampled with TEMP) vs a league (LEAGUE) on decks/pool (scripts/gen_selfplay.py).
# Env: CKPT (required), GAMES (total, default 4000), NPROC (default 16), SEED (default 3000000),
#   LEAGUE (default 'self:0.4,ygoagent:0.3,greedy:0.15,random:0.15'), TEMP (default 1.0), TIMEOUT (default 50m).
# Results: runs/<name>/{selfplay/p<k>/, selfplay.txt}
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
: "${CKPT:?set CKPT to the student best.pt}"
GAMES=${GAMES:-4000}; NPROC=${NPROC:-16}; SEED=${SEED:-3000000}
LEAGUE=${LEAGUE:-self:0.4,ygoagent:0.3,greedy:0.15,random:0.15}
TEMP=${TEMP:-1.0}; TIMEOUT=${TIMEOUT:-50m}
GP=$(( (GAMES + NPROC - 1) / NPROC ))
echo "selfplay $GAMES games / $NPROC procs; ckpt $CKPT; league $LEAGUE; temp $TEMP; cores $(nproc)" > "$OUT/selfplay.txt"
if command -v nvidia-smi >/dev/null 2>&1; then DEVICE=cuda; else DEVICE=cpu; fi
for p in $(seq 0 $((NPROC - 1))); do
  timeout "$TIMEOUT" uv run --no-sync python scripts/gen_selfplay.py --ckpt "$CKPT" --device "$DEVICE" \
    --games "$GP" --temp "$TEMP" --league "$LEAGUE" --seed $(( SEED + 100000 * p )) --threads 2 \
    --out "$OUT/selfplay/p$p" > "$OUT/selfplay-$p.txt" 2>&1 &
done
wait
grep -h "selfplay games\|Traceback\|Error" "$OUT"/selfplay-*.txt | tail -n 40 >> "$OUT/selfplay.txt"
du -sh "$OUT/selfplay" >> "$OUT/selfplay.txt"
echo done > "$OUT/DONE_SELFPLAY"
