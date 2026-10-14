#!/usr/bin/env bash
# Re-measure after the LoopGuard fix (PASS at SELECT_CHAIN never masked; NOTEBOOK 2026-10-14 21:30 UTC). User
# 2026-10-14: "Yes, fix and rerun". P raw, then P + hppo + value_p (the product), vs ygo-agent on FRESH deals
# (deal seeds 6000+; 5000+ chose the value net, F keeps 4000+), paired.
#   1. raw. Fail fast: scripts/audit/maxx_habit.py on the raw games must show the fix (Maxx fires at an exactly
#      repeated state <= 30% of fires; it was 93%), else ALERT and stop before the search arm.
#   2. hppo (honest k4 s4 h1, det-tries 32, skip, conf-skip 0.9, --value value_p); paired vs raw, principles s/ya,
#      maxx_habit.
# Env: GA_SEEDS (4 = 992 deals), VAR_CAP (130m). Usage: bash scripts/vast_guardfix.sh <name>. vast GPU only.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
P=data/cache/P/best.pt; VP=data/cache/value_p/best.pt; GS=${GA_SEEDS:-4}
[ -s "$P" ] && [ -s "$VP" ] || { alert "missing P or value_p"; exit 3; }
grep -q "KIND_PASS" src/ygo_zero/eval/loopguard.py || { alert "loopguard fix not in this tree"; exit 3; }
H="--honest 1 --det-tries 32 --det-fallback skip --horizon 1 --claim 1"
SM="--k 4 --samples 4 --conf-skip 0.9 --deal-seed0 6000"
habit() {  # $1 = variant tag
  uv run --no-sync python scripts/audit/maxx_habit.py --games "$OUT/$1" --seat s --name "$1" --out "$OUT/maxx-$1.json" \
    > "$OUT/maxx-$1.txt" 2>&1 || { alert "maxx_habit $1 failed: $(tail -1 "$OUT/maxx-$1.txt")"; return 1; }
  echo "$(date +%T) maxx $1: $(tail -1 "$OUT/maxx-$1.txt" | cut -c1-400)" >> "$OUT/PROGRESS"
}
CKPT="$P" SM_ARGS="$SM" VARIANTS="raw|--opp ygoagent --raw 1 --ga-seeds $GS" \
  PROBE_ARGS="--opp ygoagent --ga-seeds $GS $H --value $VP" PROBE_NPROC=40 VAR_CAP="${VAR_CAP:-130m}" \
  bash scripts/vast_search_match.sh "$NAME"
[ -s "$OUT/raw/summary.json" ] || ls "$OUT"/raw/games-*.jsonl > /dev/null 2>&1 || { alert "no raw games"; exit 4; }
habit raw || exit 4
uv run --no-sync python - "$OUT/maxx-raw.json" <<'PY' || { alert "fix not active in raw games: $(tail -1 "$OUT/maxx-raw.txt" | cut -c1-200)"; exit 4; }
import json, sys
a = json.load(open(sys.argv[1]))["all"]
sys.exit(0 if a["fires_at_repeated_state"] <= 0.30 else 1)
PY
CKPT="$P" SM_ARGS="$SM" VARIANTS="hppo|--opp ygoagent --ga-seeds $GS $H --value $VP" \
  PROBE_NPROC=40 VAR_CAP="${VAR_CAP:-130m}" bash scripts/vast_search_match.sh "$NAME"
uv run --no-sync python scripts/audit/paired_sm.py "$OUT/raw" "$OUT/hppo" > "$OUT/paired.txt" 2>&1 \
  || alert "paired_sm: $(tail -1 "$OUT/paired.txt")"
echo "$(date +%T) paired hppo - raw: $(sed -n 1,4p "$OUT/paired.txt" | tr '\n' ' ')" >> "$OUT/PROGRESS"
habit hppo
for v in raw hppo; do for s in s ya; do
  uv run --no-sync python scripts/audit/principles.py --name "$v-$s" --games "$OUT/$v" --seat $s \
    --out "$OUT/principles-$v-$s.json" > "$OUT/principles-$v-$s.txt" 2>&1 || alert "principles $v $s failed"
done
uv run --no-sync python - "$OUT" "$v" <<'PY' >> "$OUT/PROGRESS"
import json, sys
o, v = sys.argv[1], sys.argv[2]
s, ya = (json.load(open(f"{o}/principles-{v}-{x}.json")) for x in ("s", "ya"))
print(f"{v} behaviour: own-T1 HT games {s['lines_with_own_t1_ht']} (gate <= 1.5 x ygo-agent "
      f"{ya['lines_with_own_t1_ht']}); interrupts {s['interrupt_opp']['rate']} vs ygo-agent {ya['interrupt_opp']['rate']}")
PY
done
