#!/usr/bin/env bash
# Step 1a deck qualification on a vast box (docs/plans/2026-11-03-opp-turn-interaction.md): runs combo_pool.py over the
# decks in data/cache/combo_pool/decks.txt, one single-thread process per deck slice, processes sized by
# min(cores, RAM / 1.3 GB, decks). Usage (on the box): bash scripts/vast_combo_pool.sh <name>
set -uo pipefail
NAME=${1:?name}
cd "$(dirname "$0")/.."
OUT=runs/$NAME
mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
# CPU torch is enough here (policy forward on single positions); vast_setup.sh installs only the dev group.
uv sync --group dev --group ml -q > "$OUT/sync.txt" 2>&1
uv run --no-sync python -c "import torch" || { alert "no torch after uv sync"; exit 3; }
DECKS=$(cat "${DECKS_FILE:-data/cache/combo_pool/decks.txt}")
ND=$(tr ',' '\n' <<< "$DECKS" | grep -c .)
[ "$ND" -ge 1 ] || { alert "empty deck list"; exit 3; }
# The container's limits, not the host's: nproc and MemAvailable can show the whole host (256 cores / 489 GB on a
# box rented as 32 effective cores, 2026-11-03).
read -r Q PER < /sys/fs/cgroup/cpu.max 2>/dev/null || {  # cgroup v2, else v1 (quota -1 = none)
  Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); PER=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 1)
  [ "$Q" -gt 0 ] || Q=max; }
CORES=$(nproc); [ "$Q" != max ] && CORES=$(( Q / PER < CORES ? Q / PER : CORES ))
MEM_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
ML=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
[ "$ML" != max ] && [ $((ML / 1073741824)) -lt "$MEM_GB" ] && MEM_GB=$((ML / 1073741824))
NP=$(( CORES < MEM_GB * 10 / 13 ? CORES : MEM_GB * 10 / 13 ))
[ "$NP" -gt "$ND" ] && NP=$ND  # one deck per process when the box allows: no idle tail
# The rented core count (launcher MIN_CPUS) caps it too: the cgroup limit is not always visible (a 32-core rental
# showed nproc 256 and no quota, 2026-11-03), and RAM per process (~1 GB) must hold.
[ -n "${MAX_NP:-}" ] && [ "$NP" -gt "$MAX_NP" ] && NP=$MAX_NP
[ "$NP" -ge 1 ] || { alert "no room for a process (mem ${MEM_GB}G)"; exit 3; }
[ "${SMOKE:-0}" = 1 ] && { DECKS=$(cut -d, -f1-2 <<< "$DECKS"); ND=2; NP=2; }
echo "decks $ND procs $NP cores $CORES (nproc $(nproc)) mem ${MEM_GB}G" | tee -a "$OUT/PROGRESS"
for i in $(seq 0 $((NP - 1))); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python scripts/audit/combo_pool.py --decks "$DECKS" \
    --seeds 12 --samples 4 --hands 6 --shard $i/$NP --out "$OUT/s$i.json" > "$OUT/s$i.log" 2>&1 &
done
( while sleep 60; do { uptime; free -g | sed -n 2p; } > "$OUT/usage.txt"; \
    echo "decks done $(cat "$OUT"/s*.log | grep -c ' s)$')/$ND" >> "$OUT/PROGRESS"; done ) &
MON=$!
wait $(jobs -p | grep -v "^$MON$")
kill $MON 2>/dev/null
grep -h "ALERT\|Traceback" "$OUT"/s*.log >> "$OUT/ALERT" 2>/dev/null
N=$(python3 -c "import json,glob; print(sum(len(json.load(open(f))['rows']) for f in glob.glob('$OUT/s*.json')))")
echo "rows $N/$ND" | tee -a "$OUT/PROGRESS"
[ "$N" -eq "$ND" ] || { alert "rows $N != decks $ND"; exit 3; }
