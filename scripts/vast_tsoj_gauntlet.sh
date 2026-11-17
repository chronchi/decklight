#!/usr/bin/env bash
# tsoj gauntlet, step 1 (NOTEBOOK 2026-11-07 gauntlet pre-registration): P vs tsoj's `bot alphabeta` on a mirror book
# (every deck in both seats). Node sweep on SWEEP_SEEDS book lines per deck pair, N* chosen by
# scripts/tsoj/gauntlet_summary.py --choose, then the main gauntlet at N* on fresh seeds. CPU only. Not on the laptop.
# Env: SWEEP ("500 1000 2000 5000"), SWEEP_SEEDS (4), MAIN_SEEDS (28), CAP_SWEEP (40m per level), CAP_MAIN (100m).
# Usage: bash scripts/vast_tsoj_gauntlet.sh <name>
set -uo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/tsoj-arms/$NAME"; mkdir -p "$OUT" "runs/$NAME"
trap 'cp -r "$OUT"/. "runs/$NAME"/ 2>/dev/null' EXIT
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
progress() { echo "$*" | tee -a "$OUT/PROGRESS"; }
export OMP_NUM_THREADS=1
find bots -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null
uv sync --group dev --group ml -q || { alert "uv sync failed"; exit 3; }
. scripts/cores.sh
J=$(box_cores)
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &

echo "== toolchain $(date +%T)"
command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null || { alert "rustup failed"; exit 3; }; }
command -v cmake >/dev/null || { uv tool install -q cmake || { alert "cmake install failed"; exit 3; }; }
export YGO_AI="$HOME/yugioh_ai"
[ -d "$YGO_AI" ] || git clone -q https://github.com/tsoj/yugioh_ai "$YGO_AI" || { alert "clone failed"; exit 3; }
echo "== build $(date +%T)"
TASKSET="" CARGO_BUILD_JOBS="$J" bash scripts/tsoj/build_tsoj.sh > "$OUT/build.log" 2>&1 || { alert "build_tsoj failed: $(tail -3 "$OUT/build.log" | tr '\n' ' ')"; exit 3; }
progress "build done $(date +%T)"

B=scripts/tsoj/books
uv run --no-sync python scripts/tsoj/make_book.py --mirror --seeds "${SWEEP_SEEDS:-4}" --out $B/sweep.jsonl | head -1 | tee -a "$OUT/PROGRESS"
uv run --no-sync python scripts/tsoj/make_book.py --mirror --seeds "${MAIN_SEEDS:-28}" --seed-base 10000 --out $B/main.jsonl | head -1 | tee -a "$OUT/PROGRESS"
RT=$(free -m | awk '/Mem/{print $2}'); NP=$(( J < RT * 8 / 10 / 1300 ? J : RT * 8 / 10 / 1300 )); [ "$NP" -lt 1 ] && NP=1
echo "cores $J, ram ${RT}MB -> $NP games at once" | tee "$OUT/sizing.txt"

run() {  # run OUTNAME NODES N BOOK CAP; prints label=file:wall for gauntlet_summary
  local t0=$(date +%s)
  OUTNAME="$1" timeout "$5" bash scripts/tsoj/run_arms.sh "$NAME" p "$2" "$3" "$NP" "$4" \
    > "$OUT/run-$1.txt" 2>&1 || alert "run $1 exit $? ($(tail -2 "$OUT/run-$1.txt" | tr '\n' ' '))"
  local w=$(( $(date +%s) - t0 ))
  progress "$1 (nodes $2): ${w}s; $(uv run --no-sync python scripts/tsoj/game_stats.py "$OUT/$1.jsonl" 2>&1 | tail -1 | cut -c1-300)"
  echo "$2=$OUT/$1.jsonl:$w" >> "$OUT/specs-${1%%-*}.txt"
}
# first milestone: a short calibration run must give real duels
run cal 500 "$(( NP / 2 > 1 ? NP / 2 : 1 ))" $B/sweep.jsonl 15m
uv run --no-sync python scripts/tsoj/game_stats.py "$OUT/cal.jsonl" > /dev/null 2>&1 || { alert "calibration games are not real duels: stop"; exit 4; }
NB=$(wc -l < $B/sweep.jsonl)
for n in ${SWEEP:-500 1000 2000 5000}; do
  run "sweep-n$n" "$n" "$NB" $B/sweep.jsonl "${CAP_SWEEP:-40m}"
  progress "$(uv run --no-sync python scripts/tsoj/gauntlet_summary.py "$n=$OUT/sweep-n$n.jsonl" 2>&1 | tail -1 | cut -c1-400)"
done
uv run --no-sync python scripts/tsoj/gauntlet_summary.py $(cat "$OUT/specs-sweep.txt") --choose --out "$OUT/sweep.json" > "$OUT/sweep.txt" 2>&1 || { alert "sweep summary failed: $(tail -2 "$OUT/sweep.txt" | tr '\n' ' ')"; exit 4; }
NSTAR=$(grep -o 'N\*=[0-9]*' "$OUT/sweep.txt" | cut -d= -f2)
[ -n "$NSTAR" ] || { alert "no N* chosen"; exit 4; }
progress "N*=$NSTAR"
run "main-n$NSTAR" "$NSTAR" "$(wc -l < $B/main.jsonl)" $B/main.jsonl "${CAP_MAIN:-100m}"
uv run --no-sync python scripts/tsoj/gauntlet_summary.py "$(cat "$OUT/specs-main.txt")" --out "$OUT/main.json" > "$OUT/main.txt" 2>&1 || { alert "main summary failed"; exit 4; }
progress "main: $(tail -1 "$OUT/main.txt" | cut -c1-400)"
exit 0
