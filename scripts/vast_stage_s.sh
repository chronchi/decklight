#!/usr/bin/env bash
# Retrain stage S (docs/retrain/plan-v2.md, plan-v3.md): honest-override search (search/honest.py) on base_r3 +
# value_ppo3, the final card's hppo config otherwise (k4 s4 h1, det-tries 32, skip, conf-skip 0.9), on the final
# card's 1,488 ygo-agent deals (seeds 1000-1005), paired with the stored hppo games (shipped in data/cache/stage_s/hppo).
# Variants: ho (both seats), ho_t1 (going first, own turn 1 searched too). Then the override replication audit
# (scripts/audit/replicate_overrides.py) on 300 ho and 300 hppo overrides with 32 fresh worlds each.
# First milestone: after 150 ho games, scripts/audit/stage_s_milestone.py; if it fails (rollout cost > 1.4x hppo or
# no pairing) the search processes are killed (fail fast).
# Usage: bash scripts/vast_stage_s.sh <name>. Do NOT run on the laptop (vast GPU only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
REF=data/cache/stage_s/hppo
[ "$(ls $REF/games-*.jsonl 2>/dev/null | wc -l)" -gt 0 ] || { alert "missing hppo reference games in $REF"; exit 3; }
H="--honest 1 --det-tries 32 --det-fallback skip --value data/cache/value_ppo3/best.pt --horizon 1 --claim 1"
H="$H --ho 1 --ho-z 2 --ho-min-gap 0.05 --ho-s-max 8 --ho-step 4 --max-searches 400"
V="ho|--opp ygoagent --ga-seeds 6 $H"
V="$V;ho_t1|--opp ygoagent --ga-seeds 6 --s-seat-only 0 --own-turn1 1 $H"
( while sleep 60; do
    [ -f "$OUT/DONE" ] && break
    n=$(cat "$OUT"/ho/games-*.jsonl 2>/dev/null | wc -l)
    if [ "$n" -ge 150 ]; then
      if uv run --no-sync python scripts/audit/stage_s_milestone.py "$OUT/ho" "$REF" > "$OUT/milestone.txt" 2>&1; then
        echo "$(date +%T) $(tail -1 "$OUT/milestone.txt")" >> "$OUT/PROGRESS"
      else
        alert "$(tail -1 "$OUT/milestone.txt"); killing the search"
        touch "$OUT/KILLED"
        kill -- -"$(cat "$OUT/sm.pgid")" 2>/dev/null
      fi
      break
    fi
  done ) &
setsid bash -c 'echo $$ > "'"$OUT"'/sm.pgid"; exec env CKPT=data/cache/base_r3/best.pt SM_ARGS="--k 4 --samples 4 --conf-skip 0.9 --deal-seed0 1000" VARIANTS="'"$V"'" PROBE_ARGS="--opp ygoagent --ga-seeds 6 '"$H"'" PROBE_NPROC=40 GPU_MARGIN=1.15 VAR_CAP="${VAR_CAP:-240m}" bash scripts/vast_search_match.sh "'"$NAME"'"'
[ -f "$OUT/KILLED" ] && exit 2
# progress line every 10 min while ho runs is in the relay; here the replication audit
NP=$(grep -o -- '-> [0-9]* procs' "$OUT/sizing.txt" | grep -o '[0-9]*'); NP=${NP:-8}
for tag in ho hppo; do
  d="$OUT/ho"; [ "$tag" = hppo ] && d="$REF"
  PIDS=()
  for ((p = 0; p < NP; p++)); do
    timeout 60m uv run --no-sync python scripts/audit/replicate_overrides.py --games "$d" --n 300 --samples 32 \
      --ckpt data/cache/base_r3/best.pt --value data/cache/value_ppo3/best.pt --nproc "$NP" --proc "$p" \
      --out "$OUT/repl/$tag.jsonl" > "$OUT/repl-$tag-$p.txt" 2>&1 &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || alert "repl $tag: proc exit $?"; done
  if uv run --no-sync python scripts/audit/replicate_overrides.py --games "$d" --ckpt x --value x --out "$OUT/repl/$tag.jsonl" \
      --summarize > "$OUT/repl-$tag.txt" 2>&1; then
    echo "repl $tag $(tail -1 "$OUT/repl-$tag.txt")" >> "$OUT/PROGRESS"
  else
    alert "repl $tag summarize: $(tail -1 "$OUT/repl-$tag.txt")"
  fi
done
echo done > "$OUT/DONE_ALL"
