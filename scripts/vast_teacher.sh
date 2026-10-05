#!/usr/bin/env bash
# ygo-agent teacher games (gen_teacher.py) on a many-core box. Usage: bash scripts/vast_teacher.sh <name>
# Env: GAMES (total, default 3000), PROCS (parallel processes, default cores/2). Records → runs/<name>/teacher/p<k>/
# (one dir per process: parallel appends to one file could interleave long lines).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT/teacher"
uv sync --group dev --group gauntlet -q
N=$(nproc); P=${PROCS:-$(( N / 2 ))}; [ "$P" -lt 1 ] && P=1; G=${GAMES:-3000}; PER=$(( (G + P - 1) / P ))
echo "procs $P games/proc $PER cores $N" > "$OUT/teacher.txt"
T0=$SECONDS
for p in $(seq 0 $((P - 1))); do
  timeout 100m uv run --no-sync python scripts/gen_teacher.py --out "$OUT/teacher/p$p" --games "$PER" \
    --seed $(( p * 100000 )) --threads 2 --sample-frac 0.5 > "$OUT/teacher-$p.txt" 2>&1 &
done
wait
echo "wall $((SECONDS - T0)) s" >> "$OUT/teacher.txt"
cat "$OUT"/teacher-*.txt | grep -E "games written|Error|Traceback" >> "$OUT/teacher.txt"
find "$OUT/teacher" -name "*.jsonl" | wc -l >> "$OUT/teacher.txt"; du -sh "$OUT/teacher" >> "$OUT/teacher.txt"
echo done > "$OUT/DONE"
