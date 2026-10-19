#!/usr/bin/env bash
# Fast vs Full primer quality (NOTEBOOK 2026-10-19 pre-registration): play the 12-deck ring with the Fast product
# (raw P) and the Full product (P + hppo + value_p2, both seats), then impact-score each game dir separately.
# Env: as vast_search_match.sh (CKPT, VARIANTS, SM_ARGS, PROBE_ARGS, VAR_CAP) + VALUE (value checkpoint, must exist).
# Usage: bash scripts/vast_primer_fvf.sh <name>. Results: runs/<name>/{fast,full}/games-*.jsonl and
# runs/<name>/score-{fast,full}/{summaries.json.gz,primers.json}.
set -uo pipefail
NAME="${1:?usage: bash scripts/vast_primer_fvf.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
: "${VALUE:?set VALUE}"; [ -f "$VALUE" ] || { alert "missing $VALUE"; exit 3; }
# one variant at a time, each scored at once: a scoring failure shows after the fast games, not after the full ones
IFS=';' read -ra VS <<< "$VARIANTS"
for v in "${VS[@]}"; do
  t="${v%%|*}"
  VARIANTS="$v" bash scripts/vast_search_match.sh "$NAME" || { alert "$t games stage exited $?"; exit 3; }
  rm -f "$OUT/DONE"
  n=$(cat "$OUT/$t"/games-*.jsonl 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || { alert "$t: 0 games; scoring skipped"; exit 3; }
  echo "$(date +%T) $t: $n games recorded; scoring" >> "$OUT/PROGRESS"
  bash scripts/vast_primer_impact.sh "$NAME/score-$t" "$OUT/$t" || {
    alert "scoring $t exited $?"; cat "$OUT/score-$t/ALERT" >> "$OUT/ALERT" 2>/dev/null; exit 3; }
  grep -h "impact:" "$OUT/score-$t/PROGRESS" | tail -2 | sed "s/^/score-$t /" >> "$OUT/PROGRESS"
done
echo done > "$OUT/DONE"
