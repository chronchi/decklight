#!/usr/bin/env bash
# Full puzzle mining (web-ux-plan step 5, item 8): one mine_puzzles.py process per games file (CPU fast-mode bot, no
# CUDA), each keeping up to PER_FILE verified puzzles; the laptop merges them (scripts/merge_puzzles.py).
# Sizing: procs = min(usable cores, RAM*0.8 / 1.8 GB) (measured on the laptop: one verifier ~1.3 GB RSS).
# A file with too few games may legitimately yield 0 puzzles (exit 2); the job alerts when more than half the files fail.
# Usage: bash scripts/vast_mine_puzzles.sh <name> <games dir>...  Results: runs/<name>/shards/*.json, PROGRESS, ALERT.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage}"; shift; OUT="runs/$NAME"; mkdir -p "$OUT/shards" "$OUT/logs"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
uv sync --group dev --group ml --group gauntlet -q || { alert "uv sync failed"; exit 3; }
. scripts/cores.sh
J=$(box_cores); RTOT=$(free -m | awk '/Mem/{print $2}')
NP=$(python3 -c "print(max(1, min($J, int($RTOT*0.8/1800))))")
echo "cores $J, ram ${RTOT}MB -> $NP procs" | tee "$OUT/sizing.txt"
( while true; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: > "$OUT/jobs.txt"
for d in "$@"; do for f in "$d"/games-*.jsonl; do echo "$d $(basename "$f")" >> "$OUT/jobs.txt"; done; done
echo "$(date +%T) $(wc -l < "$OUT/jobs.txt") files on $NP procs" >> "$OUT/PROGRESS"
run_one() {
  local d=$1 f=$2 tag; tag="$(basename "$(dirname "$d")")-$(basename "$d")-${f%.jsonl}"
  if timeout "${FILE_CAP:-40m}" uv run --no-sync python scripts/mine_puzzles.py --games "$d" --pattern "$f" --max-verify 0 \
      --target "${PER_FILE:-4}" --out "$OUT/shards/$tag.json" > "$OUT/logs/$tag.txt" 2>&1; then
    echo "$(date +%T) ok $tag $(grep -o 'puzzles: [0-9]* candidates, [0-9]* verified' "$OUT/logs/$tag.txt")" >> "$OUT/PROGRESS"
  else
    echo "$(date +%T) FAIL $tag: $(tail -1 "$OUT/logs/$tag.txt")" >> "$OUT/PROGRESS"; echo "$tag" >> "$OUT/failed.txt"
  fi
}
export -f run_one; export OUT
xargs -P "$NP" -L 1 bash -c 'run_one "$0" "$1"' < "$OUT/jobs.txt"
n=$(wc -l < "$OUT/jobs.txt"); nf=$(cat "$OUT/failed.txt" 2>/dev/null | wc -l)
grep -l "Traceback" "$OUT"/logs/*.txt 2>/dev/null | head -3 | while read -r l; do alert "traceback in $l: $(grep -m1 -A0 'Error' "$l")"; done
[ "$nf" -gt $((n / 2)) ] && alert "$nf of $n files failed"
echo "$(date +%T) done: $((n - nf)) of $n files gave puzzles, $(ls "$OUT"/shards/*.json 2>/dev/null | wc -l) shard files" >> "$OUT/PROGRESS"
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
