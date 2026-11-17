#!/usr/bin/env bash
# tsoj-arms on a many-core box (NOTEBOOK 2026-11-07): P vs tsoj's `bot alphabeta` in their arbiter, arms p / p2 (the
# determinism repeat) / top@1 / rand@1, then scripts/tsoj/arms_summary.py. CPU only. Not on the laptop.
# Env: NODES (5000), SEEDS (22 book lines per deck pair -> 14*SEEDS openings), NREP (40 openings for p2), CAP (per arm, 50m).
# Usage: bash scripts/vast_tsoj_arms.sh <name>
set -uo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/tsoj-arms/$NAME"; mkdir -p "$OUT" "runs/$NAME"
# the launcher pulls runs/$NAME: copy everything there on any exit (records, logs, client logs)
trap 'cp -r "$OUT"/. "runs/$NAME"/ 2>/dev/null' EXIT
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
progress() { echo "$*" | tee -a "$OUT/PROGRESS"; }
export OMP_NUM_THREADS=1
# local bundles (git-ignored bots/*/) point at laptop paths and make bundle discovery fail on the box (run 3: every
# client exited before joining); only the tracked bots/README.md is needed
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

uv run --no-sync python scripts/tsoj/make_book.py --seeds "${SEEDS:-22}" | head -1 | tee -a "$OUT/PROGRESS"
NB=$(wc -l < scripts/tsoj/books/tsoj.jsonl)
RT=$(free -m | awk '/Mem/{print $2}'); NP=$(( J < RT * 8 / 10 / 1300 ? J : RT * 8 / 10 / 1300 )); [ "$NP" -lt 1 ] && NP=1
echo "cores $J, ram ${RT}MB -> $NP games at once; $NB openings" | tee "$OUT/sizing.txt"

run() {  # run ARM N [OUTNAME]
  local t0=$(date +%s)
  OUTNAME="${3:-}" timeout "${CAP:-50m}" bash scripts/tsoj/run_arms.sh "$NAME" "$1" "${NODES:-5000}" "$2" "$NP" \
    > "$OUT/run-${3:-$1}.txt" 2>&1 || alert "run_arms $1 ${3:-} exit $? ($(tail -2 "$OUT/run-${3:-$1}.txt" | tr '\n' ' '))"
  progress "arm ${3:-$1}: $(( $(date +%s) - t0 ))s; $(uv run --no-sync python scripts/tsoj/game_stats.py "$OUT/${3:-$1}.jsonl" 2>&1 | tail -1 | cut -c1-300)"
}
# first milestone: a short calibration run must give real duels, else stop before spending the hour
run p "$(( NP / 2 > 1 ? NP / 2 : 1 ))" cal
uv run --no-sync python scripts/tsoj/game_stats.py "$OUT/cal.jsonl" > /dev/null 2>&1 || { alert "calibration games are not real duels: stop"; exit 4; }
run p "$NB"
run p "${NREP:-40}" p2
uv run --no-sync python scripts/tsoj/arms_summary.py "$NAME" none > "$OUT/repeat.json" 2>&1
progress "p repeat: $(grep -o '"p_repeat_identical": "[^"]*"' "$OUT/repeat.json")"
grep -q p_repeat_alert "$OUT/repeat.json" && { alert "arm p not deterministic: stop"; exit 4; }
for a in ${ARMS:-top@1 rand@1}; do run "$a" "$NB"; done
uv run --no-sync python scripts/tsoj/arms_summary.py "$NAME" ${ARMS:-top@1 rand@1} --out "$OUT/summary.json" > "$OUT/summary.txt" 2>&1
RC=$?
progress "summary exit $RC: $(python3 -c "import json;d=json.load(open('$OUT/summary.json'));print({a:{k:v.get(k) for k in ('pairs','diff','ci95','sign_p','forced_share','prefix_identical')} for a,v in d['arms'].items()}, d.get('p_repeat_identical'))" 2>&1 | cut -c1-400)"
[ "$RC" -ne 0 ] && { alert "arms_summary exit $RC"; exit 4; }
exit 0
