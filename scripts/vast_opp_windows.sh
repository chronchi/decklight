#!/usr/bin/env bash
# Opponent-turn window labels for the P fine-tune (NOTEBOOK 2026-11-04 15:10): scripts/gen_opp_windows.py sharded over
# the box's cores on the TRAINING line starts. First a built-in smoke (4 starts, then the records are replayed through
# decklight.demos.dataset.iter_samples: every labelled window must come back as a kind-window sample), then the full
# run and the same replay check on everything. CPU only. Env: STARTS, SAMPLES (16), CAP (120m).
# Usage: bash scripts/vast_opp_windows.sh <name>. Do NOT run on the laptop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT/window" "$OUT/smoke/window"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
tick() { echo "$(date +%T) $1" | tee -a "$OUT/PROGRESS"; }
export OMP_NUM_THREADS=1
uv sync --group dev --group ml -q || { alert "uv sync failed"; exit 3; }
check() { # <dir>: labelled windows in the records == window samples replayed by the training loader
  uv run --no-sync python - "$1" <<'PY'
import json, sys
from pathlib import Path
from decklight.demos.dataset import iter_samples
from decklight.engine.duel import Engine
d = Path(sys.argv[1])
recs = [json.loads(x) for f in sorted(d.glob("*.jsonl")) for x in f.read_text().splitlines() if x.strip()]
want = sum(len(r["window_info"]) for r in recs)
kinds = {}
for r in recs:
    for v in r["window_info"].values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
got = sum(1 for _, _, w, m in iter_samples(Engine(), d) if m["kind"] == "window" and w > 0)
print(f"records {len(recs)}, labelled {want} {kinds}, replayed window samples {got}")
sys.exit(0 if want and got == want else 1)
PY
}
for kd in first second; do  # both start kinds (going first / the opponent's turn 2), 3 starts each
  uv run --no-sync python scripts/gen_opp_windows.py --starts "$STARTS" --samples "${SAMPLES:-16}" --max-starts 3 \
    --only-kind $kd --shard 0/61 --out "$OUT/smoke/window/$kd.jsonl" > "$OUT/smoke/gen-$kd.txt" 2>&1 \
    || { alert "smoke gen $kd: $(tail -1 "$OUT/smoke/gen-$kd.txt")"; exit 4; }
done
check "$OUT/smoke/window" > "$OUT/smoke/check.txt" 2>&1 || { alert "smoke replay check: $(tail -1 "$OUT/smoke/check.txt")"; exit 4; }
tick "smoke ok: $(tail -1 "$OUT/smoke/check.txt")"
. scripts/cores.sh
J=$(box_cores)
RT=$(free -m | awk '/Mem/{print $2}'); NP=$(( J < RT * 8 / 10 / 1100 ? J : RT * 8 / 10 / 1100 )); [ "$NP" -lt 1 ] && NP=1
NS=$(( NP * 4 ))  # small units: 4 shards per process so no core idles at the tail
echo "cores $J, ram ${RT}MB -> $NP procs, $NS shards" | tee "$OUT/sizing.txt"
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
seq 0 $((NS - 1)) | timeout "${CAP:-120m}" xargs -P "$NP" -I{} sh -c \
  "uv run --no-sync python scripts/gen_opp_windows.py --starts '$STARTS' --samples '${SAMPLES:-16}' --shard {}/$NS \
   --out '$OUT/window/s{}.jsonl' > '$OUT/window/s{}.log' 2>&1 || echo 'shard {} exit '\$? >> '$OUT/window/fail.txt'"
[ -s "$OUT/window/fail.txt" ] && alert "gen shards: $(sort "$OUT/window/fail.txt" | uniq -c | tr '\n' ' ' | cut -c1-200)"
uv run --no-sync python - "$OUT/window" "$STARTS" "$NS" <<'PY' | tee -a "$OUT/PROGRESS"
import json, sys, glob
c, missing = {}, []
for i in range(int(sys.argv[3])):
    last = [x for x in open(f"{sys.argv[1]}/s{i}.log").read().splitlines() if x.startswith("{")]
    if not last:
        missing.append(i)
    for k, v in (json.loads(last[-1]) if last else {}).items():
        c[k] = c.get(k, 0) + v
want = len({x for x in open(sys.argv[2]).read().splitlines() if x.strip()})
print("labels", json.dumps(c), f"starts {c.get('starts', 0)}/{want}, shards without a summary {missing[:10]}")
sys.exit(0 if c.get("starts") == want and not missing else 1)
PY
COV=${PIPESTATUS[0]}
check "$OUT/window" > "$OUT/check.txt" 2>&1 || { alert "replay check: $(tail -1 "$OUT/check.txt")"; exit 4; }
tick "replay check ok: $(tail -1 "$OUT/check.txt")"
[ "$COV" -ne 0 ] && { alert "not every start labelled (see PROGRESS)"; exit 4; }
[ -s "$OUT/window/fail.txt" ] && exit 4
exit 0
