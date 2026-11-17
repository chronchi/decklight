#!/usr/bin/env bash
# Seed-resampled filter of the opponent-turn window labels (scripts/filter_opp_windows.py, NOTEBOOK 2026-11-04 20:00)
# sharded over the box's cores. First a smoke (2 labels: one shard each end), then the full run, the merge into
# $OUT/window/filtered.jsonl and a replay check (labelled windows == window samples of the training loader).
# CPU only. Env: WIN_DIR (the label records), EXPECT (their non-agree label count), STARTS, VARIANTS (12), CAP (120m).
# Usage: bash scripts/vast_filter_opp.sh <name>. Do NOT run on the laptop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; rm -rf "$OUT/parts" "$OUT/window" "$OUT/smoke"; mkdir -p "$OUT/parts" "$OUT/window" "$OUT/smoke"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
tick() { echo "$(date +%T) $1" | tee -a "$OUT/PROGRESS"; }
export OMP_NUM_THREADS=1
uv sync --group dev --group ml -q || { alert "uv sync failed"; exit 3; }
FA=(--win "$WIN_DIR" --starts "$STARTS" --variants "${VARIANTS:-12}")
T0=$(date +%s)
for s in 0 $((EXPECT - 1)); do  # the first and the last label
  uv run --no-sync python scripts/filter_opp_windows.py "${FA[@]}" --shard $s/$EXPECT --max-labels 1 \
    --out "$OUT/smoke/s$s.json" > "$OUT/smoke/s$s.txt" 2>&1 || { alert "smoke shard $s: $(tail -1 "$OUT/smoke/s$s.txt")"; exit 4; }
done
uv run --no-sync python - "$OUT"/smoke/s*.json <<'PY' > "$OUT/smoke/check.txt" 2>&1 || { alert "smoke: no valid variant: $(cat "$OUT/smoke/check.txt")"; exit 4; }
import json, sys
rows = [r for f in sys.argv[1:] for r in json.load(open(f))["rows"]]
print([(r["orig"], r["start_kind"], r["valid"], r["keep"]) for r in rows])
sys.exit(0 if len(rows) == 2 and all(r["valid"] > 0 for r in rows) else 1)
PY
tick "smoke ok ($(( $(date +%s) - T0 )) s for 2 labels): $(grep -h '^1/1' "$OUT"/smoke/s*.txt | cut -c1-200 | tr '\n' ' ')"
. scripts/cores.sh
J=$(box_cores)
RT=$(free -m | awk '/Mem/{print $2}'); NP=$(( J < RT * 8 / 10 / 1100 ? J : RT * 8 / 10 / 1100 )); [ "$NP" -lt 1 ] && NP=1
NS=$(( NP * 4 )); [ "$NS" -gt "$EXPECT" ] && NS=$EXPECT  # small units so no core idles at the tail; never empty
echo "cores $J, ram ${RT}MB -> $NP procs, $NS shards" | tee "$OUT/sizing.txt"
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
seq 0 $((NS - 1)) | timeout "${CAP:-120m}" xargs -P "$NP" -I{} sh -c \
  "uv run --no-sync python scripts/filter_opp_windows.py ${FA[*]} --shard {}/$NS --out '$OUT/parts/s{}.json' \
   > '$OUT/parts/s{}.txt' 2>&1 || echo 'shard {} exit '\$? >> '$OUT/parts/fail.txt'"
XS=${PIPESTATUS[1]}
[ "$XS" -ne 0 ] && { alert "filter run exit $XS (124 = CAP ${CAP:-120m} reached)"; exit 4; }
[ -s "$OUT/parts/fail.txt" ] && { alert "filter shards: $(sort "$OUT/parts/fail.txt" | uniq -c | tr '\n' ' ' | cut -c1-200)"; exit 4; }
uv run --no-sync python scripts/filter_opp_windows.py "${FA[@]}" --merge "$OUT"/parts/s*.json \
  --expect-parts "$NS" --expect-labels "$EXPECT" \
  --out "$OUT/window/filtered.jsonl" > "$OUT/merge.txt" 2>&1 || { alert "merge: $(tail -1 "$OUT/merge.txt")"; exit 4; }
tick "filtered: $(tail -1 "$OUT/merge.txt" | cut -c1-600)"
uv run --no-sync python - "$OUT/window" <<'PY' > "$OUT/check.txt" 2>&1 || { alert "replay check: $(tail -1 "$OUT/check.txt")"; exit 4; }
import json, sys
from pathlib import Path
from decklight.demos.dataset import iter_samples
from decklight.engine.duel import Engine
d = Path(sys.argv[1])
recs = [json.loads(x) for f in sorted(d.glob("*.jsonl")) for x in f.read_text().splitlines() if x.strip()]
want = sum(r["labels"].count("window") for r in recs)
got = sum(1 for _, _, w, m in iter_samples(Engine(), d) if m["kind"] == "window" and w > 0)
print(f"records {len(recs)}, labelled {want}, replayed window samples {got}")
sys.exit(0 if want and got == want else 1)
PY
tick "replay check ok: $(tail -1 "$OUT/check.txt")"
exit 0
