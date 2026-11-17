#!/usr/bin/env bash
# Cycle-1 G4 (docs/lab/prereg/2026-11-10-master-cycle1.md): X2 vs tsoj's `bot alphabeta` at 500 nodes on the held-out
# mirror book (seeds 20000+, 28 per deck pair: the 1,568 games of X's stored eval), paired X2 - X. X is re-played on
# the same box: its stored eval (window-ppo-20261108-0940) predates the live-client obs fixes of 2026-11-10 12:40, so
# a paired comparison against it would mix the client change into the gate. The stored eval is summarised alongside
# (X-old) to show the client effect. Also the going-first turn-2 audit (prereg amendment 4) for both arms.
# CPU only (the raw agent runs in each WindBot client). Env: SEEDS (28), CAP_ARM (90m), CAL (1: calibration first).
# Usage: bash scripts/vast_c1_g4.sh <name>
set -uo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
NAME="$1"; OUT="$PWD/runs/tsoj-arms/$NAME"; mkdir -p "$OUT" "runs/$NAME"
trap 'cp -r "$OUT"/. "runs/$NAME"/ 2>/dev/null; pkill -P $$ 2>/dev/null' EXIT
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
progress() { echo "$(date +%T) $*" | tee -a "$OUT/PROGRESS"; }
export OMP_NUM_THREADS=1
G=data/cache/c1_g4
for f in $G/X/best.pt $G/X/metrics.jsonl $G/X2/best.pt $G/X2/metrics.jsonl $G/eval-X-old.jsonl; do [ -s "$f" ] || { alert "missing $f"; exit 3; }; done
cmp -s $G/X/best.pt $G/X2/best.pt && { alert "X and X2 are the same file"; exit 3; }
find bots -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null
uv sync --group dev --group ml -q || { alert "uv sync failed"; exit 3; }
. scripts/cores.sh
J=$(box_cores)
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &

progress "toolchain"
command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null || { alert "rustup failed"; exit 3; }; }
command -v cmake >/dev/null || { uv tool install -q cmake || { alert "cmake install failed"; exit 3; }; }
export YGO_AI="$HOME/yugioh_ai"
[ -d "$YGO_AI" ] || git clone -q https://github.com/tsoj/yugioh_ai "$YGO_AI" || { alert "clone failed"; exit 3; }
TASKSET="" CARGO_BUILD_JOBS="$J" bash scripts/tsoj/build_tsoj.sh > "$OUT/build.log" 2>&1 || { alert "build_tsoj failed: $(tail -3 "$OUT/build.log" | tr '\n' ' ')"; exit 3; }
progress "build done"

B=scripts/tsoj/books
uv run --no-sync python scripts/tsoj/make_book.py --mirror --seeds "${SEEDS:-28}" --seed-base 20000 --out $B/heldout.jsonl | head -1 | tee -a "$OUT/PROGRESS"
NB=$(wc -l < $B/heldout.jsonl)
RT=$(free -m | awk '/Mem/{print $2}'); NP=$(( J < RT * 8 / 10 / 1300 ? J : RT * 8 / 10 / 1300 )); [ "$NP" -lt 1 ] && NP=1
progress "cores $J, ram ${RT}MB -> $NP games at once; book $NB lines"

run() {  # run LABEL CKPT N BOOK CAP: one arm, raw agent, 500 nodes
  local t0; t0=$(date +%s); local f0; f0=$(date +%Y-%m-%dT%H:%M:%S)
  DL_AGENT=raw DL_CKPT="$PWD/$2" OUTNAME="eval-$1" timeout "$5" bash scripts/tsoj/run_arms.sh "$NAME" p 500 "$3" "$NP" "$4" \
    > "$OUT/run-$1.txt" 2>&1 || alert "run $1 exit $? ($(tail -2 "$OUT/run-$1.txt" | tr '\n' ' '))"
  echo "$1=$OUT/eval-$1.jsonl:$(( $(date +%s) - t0 ))" > "$OUT/spec-$1.txt"
  echo "$f0,$(date +%Y-%m-%dT%H:%M:%S)" > "$OUT/window-$1.txt"
  progress "$1: $(( $(date +%s) - t0 ))s; $(uv run --no-sync python scripts/tsoj/game_stats.py "$OUT/eval-$1.jsonl" 2>&1 | tail -1 | cut -c1-300)"
}
# first milestone: a short X2 calibration must give real duels (an observable-history checkpoint raises if the
# client does not build obs_history)
if [ "${CAL:-1}" = 1 ]; then
  run cal $G/X2/best.pt "$(( NP / 2 > 1 ? NP / 2 : 1 ))" $B/heldout.jsonl 15m
  uv run --no-sync python scripts/tsoj/game_stats.py "$OUT/eval-cal.jsonl" > /dev/null 2>&1 || { alert "calibration games are not real duels: stop"; exit 4; }
  grep -l "observable-history checkpoint" "$OUT"/logs/*.log 2>/dev/null | head -1 | grep -q . && { alert "X2 client lacks obs_history: stop"; exit 4; }
fi
run X2 $G/X2/best.pt "$NB" $B/heldout.jsonl "${CAP_ARM:-90m}"
run X $G/X/best.pt "$NB" $B/heldout.jsonl "${CAP_ARM:-90m}"
cmp -s "$OUT/eval-X.jsonl" "$OUT/eval-X2.jsonl" && { alert "X and X2 records identical"; exit 4; }
uv run --no-sync python scripts/tsoj/gauntlet_summary.py "$(cat "$OUT/spec-X2.txt")" "$(cat "$OUT/spec-X.txt")" \
  "X-old=$G/eval-X-old.jsonl" --paired X --out "$OUT/eval.json" > "$OUT/eval.txt" 2>&1 \
  || { alert "eval summary failed: $(tail -2 "$OUT/eval.txt" | tr '\n' ' ')"; exit 4; }
progress "G4: $(grep -h 'paired\|"X2"\|"X"' "$OUT/eval.txt" | cut -c1-600 | tr '\n' ' ')"
python3 scripts/tsoj/turn2_interrupts.py "X2=$OUT/logs@$(cat "$OUT/window-X2.txt")" "X=$OUT/logs@$(cat "$OUT/window-X.txt")" \
  --out "$OUT/turn2.json" > "$OUT/turn2.txt" 2>&1 || alert "turn2 audit failed: $(tail -2 "$OUT/turn2.txt" | tr '\n' ' ')"
progress "turn2: $(cut -c1-600 "$OUT/turn2.txt" | tr '\n' ' ')"
exit 0
