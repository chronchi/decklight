#!/usr/bin/env bash
# A3c played gate on a vast box (launcher: scripts/launch/launch-a3c.sh; plan and check: scripts/app/a3c_gate.py).
# Runs after scripts/vast_setup.sh (uv venv, core, Python ext). Builds the Rust product_games here (the laptop binary
# needs a newer libstdc++ than the image), then runs the planned units in order until RUN_MIN minutes have passed.
#   bash scripts/app/vast_a3c.sh <name>
# Env: RUN_MIN (unit time, minutes, from the script's start), PAIRS (150), UNITS (5), SEED (61000), NPROC (nproc),
#      SMOKE=1 (2 pairs, 1 unit each).
set -uo pipefail
N=$1
OUT=runs/$N
mkdir -p "$OUT/u"
START=$(date +%s)
export DEADLINE=$((START + ${RUN_MIN:-60} * 60))
# Processes: min(cores, available RAM / 0.9 GB). Measured on the laptop 2026-11-02: product_games 0.50 GB RSS,
# product_games_py 0.84 GB, both ~4 min per game on one core.
# A container's nproc can list every host core; its cgroup quota (cpu.max) is what it may use.
CORES=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ] && read -r Q P < /sys/fs/cgroup/cpu.max && [ "$Q" != max ]; then
  [ $(( Q / P )) -lt "$CORES" ] && CORES=$(( Q / P ))
fi
MEMG=$(free -g | awk '/Mem/{print $7}')
NP=${NPROC:-$(( CORES < MEMG * 10 / 9 ? CORES : MEMG * 10 / 9 ))}
alert() { echo "$*" >> "$OUT/ALERT"; echo "ALERT $*"; }
( while true; do echo "$(date +%T) load $(cut -d' ' -f1 /proc/loadavg) mem_used $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE=$!

# Rust toolchain and the product_games binary.
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > "$OUT/rustup.log" 2>&1 || { alert "rustup failed"; exit 1; }
fi
. "$HOME/.cargo/env"
( cd rust && cargo build --release -j "$NP" -p decklight-search --bin product_games ) > "$OUT/build.log" 2>&1 \
  || { alert "cargo build failed (tail: $(tail -3 "$OUT/build.log" | tr '\n' ' '))"; exit 1; }
# The Python oracle needs onnxruntime and (CPU) torch: vast_setup.sh syncs only the dev group. The locked versions
# are the laptop's, so both sides load the same libonnxruntime (2026-11-02: the first launch died here).
uv sync --group dev --group onnx --group ml -q > "$OUT/sync.log" 2>&1 || { alert "uv sync onnx/ml failed (tail: $(tail -3 "$OUT/sync.log" | tr '\n' ' '))"; exit 1; }
uv run --no-sync python -c "import onnxruntime, torch, decklight.model.onnx_export" > "$OUT/imports.log" 2>&1 \
  || { alert "python oracle imports fail (tail: $(tail -2 "$OUT/imports.log" | tr '\n' ' '))"; exit 1; }
ORT=$(ls .venv/lib/python3*/site-packages/onnxruntime/capi/libonnxruntime.so.* 2>/dev/null | head -1)
[ -n "$ORT" ] || { alert "no onnxruntime library in .venv"; exit 1; }
export ORT_DYLIB_PATH=$PWD/$ORT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OUT
export SEED=${SEED:-61000}
echo "built in $(( $(date +%s) - START ))s; ORT $ORT; $NP processes; deadline in $(( DEADLINE - $(date +%s) ))s"

if [ "${SMOKE:-0}" = 1 ]; then
  uv run --no-sync python scripts/app/a3c_gate.py plan --pairs 2 --units-per-pair 1 --seed "$SEED" --out "$OUT/units.txt"
else
  uv run --no-sync python scripts/app/a3c_gate.py plan --pairs "${PAIRS:-150}" --units-per-pair "${UNITS:-5}" \
    --seed "$SEED" --out "$OUT/units.txt"
fi

run_unit() {
  local kind=$1 pair=$2 k=$3
  local t="${kind}_${pair/:/__}_$k" left=$(( DEADLINE - $(date +%s) ))
  if [ "$left" -lt 240 ]; then echo 200 > "$OUT/u/$t.rc"; return 0; fi  # 200 = not started (deadline)
  local off=$(( 2 * k ))
  if [ "$kind" = rs ]; then
    timeout "$left" rust/target/release/product_games --pairs "$pair" --games 2 --game-offset "$off" --seed "$SEED" \
      --choice-seed "$SEED" --ort-threads 1 --replay-threads 1 --out "$OUT/u/$t.jsonl.gz" > "$OUT/u/$t.log" 2>&1
  else
    timeout "$left" uv run --no-sync python scripts/app/product_games_py.py --pairs "$pair" --games 2 --game-offset "$off" \
      --seed "$SEED" --choice-seed "$SEED" --out "$OUT/u/$t.jsonl.gz" > "$OUT/u/$t.log" 2>&1
  fi
  local rc=$?
  echo "$rc" > "$OUT/u/$t.rc"
  [ "$rc" = 0 ] || [ "$rc" = 124 ] || echo "unit $t exit $rc: $(tail -2 "$OUT/u/$t.log" | tr '\n' ' ')" >> "$OUT/ALERT"
}
export -f run_unit

# Fail fast: compare finished pairs every 2 minutes (PROGRESS lines; ALERT on any difference or crash).
( while sleep 120; do uv run --no-sync python scripts/app/a3c_gate.py check --run "$OUT" --progress > /dev/null 2>&1; done ) &
CHECKER=$!
xargs -P "$NP" -L 1 bash -c 'run_unit "$@"' _ < "$OUT/units.txt"
kill "$CHECKER" "$USAGE" 2>/dev/null
uv run --no-sync python scripts/app/a3c_gate.py check --run "$OUT" > "$OUT/gate.txt" 2>&1
RC=$?
cat "$OUT/gate.txt"
# Keep the comparison inputs small: the units' dumps stay (they are the evidence); logs of passing units go.
for f in "$OUT"/u/*.rc; do [ "$(cat "$f")" = 0 ] && rm -f "${f%.rc}.log"; done
exit $RC
