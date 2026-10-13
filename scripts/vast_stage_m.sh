#!/usr/bin/env bash
# Retrain stage M, split mode (docs/retrain/plan-v3.md §Stage M; user 2026-10-12 "Agreed with 1 and 2"): verify muse's
# turn plans on a many-core box from the laptop's LLM cache (data/cache/llm, shipped; no OpenCode key here, the LLM never
# runs on vast). The laptop sweep verified ~4 min of one core per state and took ~7 h per 10-deck chunk (NOTEBOOK
# 2026-10-12 21:00).
# Work: NEW_GROUPS (a --deck-groups file: chunks verified in full) and, optionally, OLD_GROUPS + IDS (earlier chunks: only the
# listed states, i.e. those whose first answer came after their chunk had run). Every process samples every chunk and
# verifies its 1/NP share of the states (line-id hash), so no core idles at a per-chunk tail.
# Output: runs/<name>/turns/<bucket>/<deck>.p<i>.jsonl (merged into data/demos/turns on the laptop), the repair prompts
# that had no cached answer in runs/<name>/pending.jsonl (the laptop answers them with --llm-only --prompts, then a
# second round with the same NEW_GROUPS/IDS finishes the repairs; states already verified are skipped).
# First milestone: the first process to finish must have verified > 0 of > 10 states, else stop.
# Env: NEW_GROUPS, OLD_GROUPS, IDS, BUDGET (8000), CAP (150m). Usage: bash scripts/vast_stage_m.sh <name>. Vast only.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT/logs"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export YGO_STRICT_VERIFY=1 OMP_NUM_THREADS=1
uv sync --group dev -q
. scripts/cores.sh
J=$(box_cores); [ "$(env -u OMP_NUM_THREADS nproc)" -ge 8 ] && [ "$J" -lt 4 ] && alert "only $J usable cores detected"
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) \
  > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }
[ -s "${NEW_GROUPS:-}" ] || [ -s "${IDS:-}" ] || { alert "nothing to do: NEW_GROUPS and IDS unset or empty"; exit 3; }
[ "$(ls data/cache/llm | head -1)" ] || { alert "no LLM cache shipped"; exit 3; }
[ "$(ls data/strategy 2>/dev/null | head -1)" ] || { alert "no data/strategy shipped (prompts would not match the cache)"; exit 3; }
# RAM: a process holds the 1,445 sw-v games plus one engine (360 MB RSS measured on the laptop); 600 MB each + 20%
RTOT=$(free -m | awk '/Mem/{print $2}'); NP=$(( J < RTOT * 8 / 10 / 600 ? J : RTOT * 8 / 10 / 600 )); [ "$NP" -lt 1 ] && NP=1
echo "cores $J, ram ${RTOT}MB -> $NP procs" | tee "$OUT/sizing.txt"
A=(--games 'data/demos/sw-v/*.jsonl' --per-bucket 4 --budget "${BUDGET:-8000}" --cache-only --out-dir data/demos/turns)
run_proc() {  # one process slot: its NEW_GROUPS share, then its IDS share; each with the job's time cap
  local p=$1
  if [ -s "${NEW_GROUPS:-}" ]; then
    timeout "${CAP:-150m}" uv run --no-sync python scripts/plan_turns.py "${A[@]}" --deck-groups "$NEW_GROUPS" --proc "$p" \
      --nproc "$NP" --pending-out "$OUT/pending/p$p.jsonl" > "$OUT/logs/new-$p.txt" 2>&1 || echo "new-$p exit $?" >> "$OUT/logs/fail-$p.txt"
  fi
  if [ -s "${IDS:-}" ] && [ -s "${OLD_GROUPS:-}" ]; then
    timeout "${CAP:-150m}" uv run --no-sync python scripts/plan_turns.py "${A[@]}" --deck-groups "$OLD_GROUPS" --ids "$IDS" \
      --proc "$p" --nproc "$NP" --pending-out "$OUT/pending/p$p.jsonl" > "$OUT/logs/old-$p.txt" 2>&1 \
      || echo "old-$p exit $?" >> "$OUT/logs/fail-$p.txt"
  fi
}
mkdir -p "$OUT/pending"
PIDS=()
for p in $(seq 0 $((NP - 1))); do run_proc "$p" & PIDS+=($!); done
# fail fast: a process whose first answers are not in the cache (prompts differ from the laptop's: 2026-10-12 stage-m
# 2250, $X lost) stops the whole job at once. First milestone: the first finished log must show > 0 verified.
stop_all() { for pid in "${PIDS[@]}"; do pkill -TERM -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; done; }
( while :; do
    bad=$(grep -l "ALERT plan_turns" "$OUT"/logs/*.txt 2>/dev/null | head -1)
    if [ -n "$bad" ]; then alert "$(grep -h "ALERT plan_turns" "$bad" | head -1) ($bad); stopping"; stop_all; break; fi
    f=$(grep -l "plan_turns: verified=" "$OUT"/logs/new-*.txt "$OUT"/logs/old-*.txt 2>/dev/null | head -1)
    if [ -n "$f" ]; then
      v=$(grep -h "plan_turns: verified=" "$f" | tail -1); tick "first proc: $v ($f)"
      n=${v#*verified=}; [ "${n%%/*}" = 0 ] && [ "${n#*/}" -gt 10 ] && alert "first proc verified 0 of ${n#*/}: $(grep -h cache-only "$f")"
      break
    fi
    sleep 15
  done ) &
for pid in "${PIDS[@]}"; do wait "$pid"; done
cat "$OUT"/logs/fail-*.txt > "$OUT/logs/fail.txt" 2>/dev/null
[ -s "$OUT/logs/fail.txt" ] && alert "failed procs: $(tr '\n' ' ' < "$OUT/logs/fail.txt" | cut -c1-300)"
cat "$OUT"/pending/p*.jsonl > "$OUT/pending.jsonl" 2>/dev/null
grep -h "^ALERT" "$OUT"/logs/*.txt >> "$OUT/ALERT" 2>/dev/null
mkdir -p "$OUT/turns"
for f in data/demos/turns/*/*.p*.jsonl; do [ -s "$f" ] && { mkdir -p "$OUT/turns/$(basename "$(dirname "$f")")"; cp "$f" "$OUT/turns/$(basename "$(dirname "$f")")/"; }; done
cp data/demos/turns/report.p*.json "$OUT/logs/" 2>/dev/null
NV=$(cat "$OUT"/turns/*/*.jsonl 2>/dev/null | wc -l); NS=$(grep -h "states to plan" "$OUT"/logs/*.txt | awk '{s+=$2} END {print s+0}')
NPEND=$(wc -l < "$OUT/pending.jsonl" 2>/dev/null || echo 0)
tick "verified $NV of $NS states; $NPEND repair prompts pending"
[ "$NV" -gt 0 ] || alert "0 lines verified"
kill "$USAGE_PID" 2>/dev/null
echo done > "$OUT/DONE_ALL"
