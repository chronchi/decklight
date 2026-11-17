#!/usr/bin/env bash
# Continuation-bias check (docs/plans/2026-11-03-opp-turn-interaction.md, step 1: "after our bot breaks a line, does P
# continue the opponent's turn much worse than search would?"): `line_eval.py --bias` sharded over the box's cores.
# Each shard writes its rows after every break (a time-capped shard keeps what it did). CPU only.
# Env: STARTS (line starts file), CAP (per-shard time cap, 90m), CKPT (our policy; default P).
# Usage: bash scripts/vast_line_bias.sh <name>. Do NOT run on the laptop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT/bias"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export OMP_NUM_THREADS=1  # one thread per shard process (as stage M)
uv sync --group dev --group ml -q || { alert "uv sync failed"; exit 3; }  # ml: CPU torch (pytorch-cpu index)
uv run --no-sync python -c "import torch" || { alert "no torch"; exit 3; }
. scripts/cores.sh
J=$(box_cores)
# RAM: one process = P + value net + one engine + search (~900 MB measured for line_eval shards on the step-2 smoke)
RT=$(free -m | awk '/Mem/{print $2}'); NP=$(( J < RT * 8 / 10 / 900 ? J : RT * 8 / 10 / 900 )); [ "$NP" -lt 1 ] && NP=1
echo "cores $J, ram ${RT}MB -> $NP procs" | tee "$OUT/sizing.txt"
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
PIDS=()
for i in $(seq 0 $((NP - 1))); do
  timeout "${CAP:-90m}" uv run --no-sync python scripts/audit/line_eval.py --starts "$STARTS" --bias ${CKPT:+--ckpt "$CKPT"} \
    --shard "$i/$NP" --out "$OUT/bias/s$i.json" > "$OUT/bias/s$i.txt" 2>&1 || echo "shard $i exit $?" >> "$OUT/bias/fail.txt" &
  PIDS+=($!)
done
wait "${PIDS[@]}"
uv run --no-sync python - "$OUT" <<'PY' | tee -a "$OUT/PROGRESS"
import json, statistics, sys
from pathlib import Path
rows, games = [], 0
for f in sorted(Path(sys.argv[1], "bias").glob("s*.json")):
    d = json.loads(f.read_text()); rows += d["rows"]; games += d["games"]
m = lambda k: round(statistics.mean(r[k] for r in rows), 3) if rows else None
print(json.dumps({"games": games, "breaks": len(rows), **{k: m(k) for k in ("p_field", "s_field", "p_lost", "s_lost", "search_s")},
                  "s_better_field": sum(r["s_field"] > r["p_field"] for r in rows),
                  "p_better_field": sum(r["p_field"] > r["s_field"] for r in rows)}))
PY
[ -s "$OUT/bias/fail.txt" ] && alert "bias shards: $(sort "$OUT/bias/fail.txt" | uniq -c | tr '\n' ' ' | cut -c1-200)"
grep -q '"breaks": 0' "$OUT/PROGRESS" && alert "no break found in any shard"
true
