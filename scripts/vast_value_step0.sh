#!/usr/bin/env bash
# Value step 0 (scripts/audit/value_step0.py, NOTEBOOK 2026-11-04 23:40): Q(search) - Q(P) by the product value at the
# end of the line's turn, at every non-agree window label, sharded over the box's cores. Smoke first (first and last
# label, each must give a dQ), then the full run. The analysis against the filter's outcomes runs on the laptop.
# CPU only. Env: WIN_DIR, EXPECT (non-agree labels), STARTS, VARIANTS (4), CAP (60m).
# Usage: bash scripts/vast_value_step0.sh <name>. Do NOT run on the laptop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; rm -rf "$OUT/parts" "$OUT/smoke"; mkdir -p "$OUT/parts" "$OUT/smoke"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
tick() { echo "$(date +%T) $1" | tee -a "$OUT/PROGRESS"; }
export OMP_NUM_THREADS=1
uv sync --group dev --group ml -q || { alert "uv sync failed"; exit 3; }
FA=(--win "$WIN_DIR" --starts "$STARTS" --variants "${VARIANTS:-4}")
T0=$(date +%s)
for s in 0 $((EXPECT - 1)); do
  uv run --no-sync python scripts/audit/value_step0.py "${FA[@]}" --shard $s/$EXPECT --max-labels 1 \
    --out "$OUT/smoke/s$s.json" > "$OUT/smoke/s$s.txt" 2>&1 || { alert "smoke shard $s: $(tail -1 "$OUT/smoke/s$s.txt")"; exit 4; }
done
uv run --no-sync python - "$OUT"/smoke/s*.json <<'PY' > "$OUT/smoke/check.txt" 2>&1 || { alert "smoke: no dQ: $(cat "$OUT/smoke/check.txt")"; exit 4; }
import json, sys
rows = [r for f in sys.argv[1:] for r in json.load(open(f))["rows"]]
print([(r["orig"], r["start_kind"], r["valid"], r["dq"], r["q_search"], r["q_p"]) for r in rows])
sys.exit(0 if len(rows) == 2 and all(r["dq"] is not None for r in rows) else 1)
PY
tick "smoke ok ($(( $(date +%s) - T0 )) s for 2 labels): $(cut -c1-300 "$OUT/smoke/check.txt")"
. scripts/cores.sh
J=$(box_cores)
RT=$(free -m | awk '/Mem/{print $2}'); NP=$(( J < RT * 8 / 10 / 1400 ? J : RT * 8 / 10 / 1400 )); [ "$NP" -lt 1 ] && NP=1
NS=$(( NP * 4 )); [ "$NS" -gt "$EXPECT" ] && NS=$EXPECT
echo "cores $J, ram ${RT}MB -> $NP procs, $NS shards" | tee "$OUT/sizing.txt"
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
seq 0 $((NS - 1)) | timeout "${CAP:-60m}" xargs -P "$NP" -I{} sh -c \
  "uv run --no-sync python scripts/audit/value_step0.py ${FA[*]} --shard {}/$NS --out '$OUT/parts/s{}.json' \
   > '$OUT/parts/s{}.txt' 2>&1 || echo 'shard {} exit '\$? >> '$OUT/parts/fail.txt'"
XS=${PIPESTATUS[1]}
[ "$XS" -ne 0 ] && { alert "step0 run exit $XS (124 = CAP ${CAP:-60m} reached)"; exit 4; }
[ -s "$OUT/parts/fail.txt" ] && { alert "step0 shards: $(sort "$OUT/parts/fail.txt" | uniq -c | tr '\n' ' ' | cut -c1-200)"; exit 4; }
N=$(ls "$OUT"/parts/s*.json | wc -l); [ "$N" -eq "$NS" ] || { alert "step0: $N of $NS shard files"; exit 4; }
tick "step0 done: $(python3 -c "import json,glob; r=[x for f in glob.glob('$OUT/parts/s*.json') for x in json.load(open(f))['rows']]; print(len(r), 'rows', sum(x['dq'] is not None for x in r), 'with dQ')")"
exit 0
