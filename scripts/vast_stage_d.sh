#!/usr/bin/env bash
# Retrain stage D (docs/retrain/plan-v1.md §Stage D, plan-v2/v3; user green light 2026-10-11): strict re-verification
# (YGO_STRICT_VERIFY=1: the verifier never tries unnamed chain activations) of every stored LLM plan, then hand-variant
# expansion of the strict records. No LLM calls: the plans are in data/demos/<src>/lines/.
# Eval-only decks (decks/pool/HOLDOUT, kewl-tune, decks/newcards) are removed before verification (plan v2 stage D).
# Search-improved lines (planned*/improved) are not re-verified here: they are re-derived from the strict records
# (or dropped) at stage D's shard build. Inferred-label weights and the per-line hand-trap drops are applied later by build_bc_shards, not here.
# Sizing: one verify/expand pool of J workers (cgroup cores) over each source's whole line set, so no core idles at a
# per-deck tail. First milestone: the first source with lines must finish verifying (totals line), else stop.
# Only the lines strict mode can change are re-verified: those listed per source in JUNK (scripts/audit/strict_junk.py,
# default data/cache/audit/strict_junk.json); JUNK=none re-verifies every line. Their old records and expansions are
# replaced by these at the shard build; all other stored records are what the strict search finds anyway.
# Env: JUNK, SRCS (sources), BUDGET (2000), BUDGET_LONG (10000), VARIANTS (8).
# Usage: bash scripts/vast_stage_d.sh <name>. Results in runs/<name>/<src>/{verified,expanded}/ (pulled back).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export YGO_STRICT_VERIFY=1
SRCS="${SRCS:-planned-s4 guides planned claude planned-s3 planned-s2 planned-s1}"
# guides: lines in data/demos/lines/, verified against each guide's own decklist (data/demos/decklists)
ldir() { [ "$1" = guides ] && echo data/demos/lines || echo "data/demos/$1/lines"; }
vdir() { [ "$1" = guides ] && echo data/demos/verified || echo "data/demos/$1/verified"; }
JUNK=${JUNK:-data/cache/audit/strict_junk.json}; BUDGET=${BUDGET:-2000}; BUDGET_LONG=${BUDGET_LONG:-10000}; VARIANTS=${VARIANTS:-8}
[ "${SKIP_SYNC:-0}" = 1 ] || uv sync --group dev -q
. scripts/cores.sh
J=$(box_cores); [ "$J" -lt 2 ] && J=2; J=${JOBS:-$J}
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) \
  > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
echo "cores $J" > "$OUT/machine.txt"
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }
first=1
for src in $SRCS; do
  d="$OUT/$src"; mkdir -p "$d/verified" "$d/expanded"
  # all lines of the source in one file, eval-only decks removed (asserted below)
  uv run --no-sync python - "$(ldir "$src")" "$d/lines-train.jsonl" "$JUNK" "$src" <<'EOF' > "$d/filter.txt" 2>&1 \
    || { alert "$src: line filter failed: $(tail -1 "$d/filter.txt")"; continue; }
import json, sys
from pathlib import Path
from decklight.model.bc_data import eval_only_decks
hold, new = eval_only_decks()
only = None if sys.argv[3] == "none" else set(json.loads(Path(sys.argv[3]).read_text())[sys.argv[4]]["lines"])
keep, drop = [], 0
for f in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    for raw in f.read_text().splitlines():
        if not raw.strip():
            continue
        deck = json.loads(raw)["deck"].split("+")[0]
        if deck in hold or deck in new:
            drop += 1
        elif only is None or json.loads(raw)["id"] in only:
            keep.append(raw)
Path(sys.argv[2]).write_text("".join(x + "\n" for x in keep))
print(f"{len(keep)} train lines to verify, {drop} eval-only lines removed")
sys.exit(0 if keep or only is not None else 1)
EOF
  [ -s "$d/lines-train.jsonl" ] || { tick "$src: no lines to re-verify"; continue; }
  GARGS=(); [ "$src" = guides ] && GARGS=(--deck-from-guides data/demos/decklists)
  uv run --no-sync python scripts/verify_lines.py --lines "$d/lines-train.jsonl" "${GARGS[@]}" --seeds 0 --budget "$BUDGET" \
    --budget-long "$BUDGET_LONG" --jobs "$J" --line-timeout 300 --out "$d/verified-all.jsonl" > "$d/verify.txt" 2>&1 \
    || alert "$src: verify_lines exit $?: $(tail -1 "$d/verify.txt")"
  # split by deck so expand_lines and build_bc_shards see the usual <src>/verified/<deck>.jsonl layout
  uv run --no-sync python - "$d/verified-all.jsonl" "$d/verified" <<'EOF' >> "$d/filter.txt" 2>&1
import json, sys
from collections import defaultdict
from pathlib import Path
by = defaultdict(list)
p = Path(sys.argv[1])
for raw in (p.read_text().splitlines() if p.exists() else []):
    if raw.strip():
        by[json.loads(raw)["line_id"].split("/")[0]].append(raw)
for deck, rows in by.items():
    (Path(sys.argv[2]) / f"{deck}.jsonl").write_text("".join(r + "\n" for r in rows))
print(f"{sum(map(len, by.values()))} strict records over {len(by)} decks")
EOF
  n_v=$(cat "$d"/verified/*.jsonl 2>/dev/null | wc -l)
  n_old=$(cat "$(vdir "$src")"/*.jsonl 2>/dev/null | wc -l)
  tick "$src verified $n_v strict records (stored non-strict: $n_old; $(head -1 "$d/filter.txt"))"
  if [ "$first" = 1 ] && ! grep -q "^totals:" "$d/verify.txt"; then
    alert "$src: verify_lines did not finish at the first milestone; stopping"; exit 2
  fi
  first=0
  [ "$n_v" -gt 0 ] || { alert "$src: 0 strict records"; continue; }
  uv run --no-sync python scripts/expand_lines.py --verified "$d/verified" --lines-dir "$(ldir "$src")" \
    --out "$d/expanded" --variants "$VARIANTS" --jobs "$J" > "$d/expand.txt" 2>&1 \
    || alert "$src: expand_lines exit $?: $(tail -1 "$d/expand.txt")"
  tick "$src expanded $(cat "$d"/expanded/*.jsonl 2>/dev/null | wc -l) records"
done
kill "$USAGE_PID" 2>/dev/null
echo done > "$OUT/DONE_ALL"
