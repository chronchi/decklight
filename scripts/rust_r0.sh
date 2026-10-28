#!/usr/bin/env bash
# R0 (docs/plans/2026-10-27-experiments-and-rust.md): tsoj/ygopro-core-rust vs its C++ oracle on our pool decks.
# Needs the port built at $RS (cargo build --release -p ygo-diff with YGO_CARDS_ONLY = our pool's codes) and its oracle
# (oracle/build.sh), plus $ROOTD with external/{BabelCDB,CardScripts} -> third_party and an empty lua_fixes/.
# Plays PAIRS random ordered pool-deck pairs (seeded), GAMES games each, and prints one line per pair + totals.
# Usage: PAIRS=40 GAMES=5 bash scripts/rust_r0.sh   (laptop, one core per ygo-diff)
set -uo pipefail
cd "$(dirname "$0")/.."
RS=${RS:-data/cache/rust/ygopro-core-rust}; ROOTD=$(realpath "${ROOTD:-data/cache/rust/root}")
BIN=$RS/target/release/ygo-diff
[ -x "$BIN" ] && [ -f "$RS/target/oracle/libocgcore.so" ] || { echo "build the port and its oracle first" >&2; exit 3; }
mapfile -t PAIRS_L < <(python3 - "${PAIRS:-40}" <<'EOF'
import random, sys
from pathlib import Path
d = sorted(Path("decks/pool").glob("*.ydk"))
r = random.Random(2710)
for _ in range(int(sys.argv[1])):
    a, b = r.sample(d, 2)
    print(f"{a.resolve()},{b.resolve()}")
EOF
)
ok=0; n=0; fails=0
for p in "${PAIRS_L[@]}"; do
  line=$(cd "$RS" && YGO_ROOT="$ROOTD" timeout 600 taskset -c 0-3 ./target/release/ygo-diff --decks "$p" \
         --games "${GAMES:-5}" --seed 1 2>&1 | grep -E "games identical|mismatch|panicked" | tail -1)
  echo "$(basename "${p%%,*}" .ydk) vs $(basename "${p##*,}" .ydk): ${line:-no result}"
  m=$(echo "$line" | grep -oE "^[0-9]+/[0-9]+ games identical") || true
  if [ -n "$m" ]; then i=${m%%/*}; t=${m#*/}; t=${t%% *}; ok=$((ok + i)); n=$((n + t)); else fails=$((fails + 1)); fi
done
echo "R0 total: $ok/$n games identical; $fails pairs without a result"
[ "$fails" = 0 ] && [ "$n" -gt 0 ]
