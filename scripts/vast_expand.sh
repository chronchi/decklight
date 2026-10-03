#!/usr/bin/env bash
# CPU-box sweep: verify Claude's Kewl Tune lines, then expand the claude + planned demo sources (8 hand variants each).
# Results go to runs/$1/ (pulled back by the launcher).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
OUT="runs/$1"
# Size from the cgroup quota (cpu_cores_effective), not nproc: hosts expose all threads but rent a share.
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
if [ "$Q" = "max" ]; then J=$(nproc); else J=$(( Q / P )); fi
[ "$J" -lt 2 ] && J=2
mkdir -p "$OUT"
{ nproc; cat /sys/fs/cgroup/cpu.max; echo "jobs $J"; } > "$OUT/machine.txt" 2>&1
T0=$SECONDS
uv run python scripts/verify_lines.py --lines data/demos/claude/lines/kewl-tune.jsonl --ydk decks/pool/kewl-tune.ydk --seeds 0 \
  --budget 8000 --budget-long 15000 --jobs "$J" --line-timeout 300 --out data/demos/claude/verified/kewl-tune.jsonl > "$OUT/verify-kewl-tune.txt" 2>&1
cp data/demos/claude/verified/kewl-tune.jsonl "$OUT/" 2>/dev/null
for src in claude planned; do
  uv run python scripts/expand_lines.py --verified "data/demos/$src/verified" --lines-dir "data/demos/$src/lines" \
    --out "$OUT/$src-expanded" --variants 8 --jobs "$J" > "$OUT/expand-$src.txt" 2>&1
done
echo "wall $((SECONDS - T0)) s" > "$OUT/wall.txt"
echo done > "$OUT/DONE"
