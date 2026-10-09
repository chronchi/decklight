#!/usr/bin/env bash
# RL checkpoints vs ygo-agent on the in-run eval schedule, every game saved (why did rl5 lose to ygo-agent? 2026-10-08).
# Usage: bash scripts/vast_ga_compare.sh <name> "<tag=path[:expected seed-0 wr]> ...". One process per checkpoint in
# parallel (each --threads/--ga-workers = cores / n). Results: runs/<name>/<tag>/games.jsonl + games/*/state.json.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; SPECS="$2"; SEEDS="${SEEDS:-3}"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
if [ "$Q" = "max" ]; then J=$(nproc); else J=$(( Q / P )); fi
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  sleep 10
done
DEV=cpu; uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" && DEV=cuda
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -m | awk '/Mem/{print $3}')"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
UP=$!
N=$(echo $SPECS | wc -w); T=$(( J / N )); [ "$T" -lt 2 ] && T=2
pids=()
for spec in $SPECS; do
  tag=${spec%%=*}; rest=${spec#*=}; path=${rest%%:*}; exp=""; [ "$rest" != "$path" ] && exp="--expect $tag=${rest#*:}"
  # shellcheck disable=SC2086
  uv run --no-sync python scripts/gauntlet_games.py --ckpts "$tag=$path" $exp --deck-dir decks/gauntlet --ring 4 \
    --seeds "$SEEDS" --out "$OUT/$tag" --device "$DEV" --threads "$T" --ga-workers "$T" > "$OUT/$tag.txt" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || alert "a gauntlet_games process exited $?"; done
for spec in $SPECS; do
  tag=${spec%%=*}; n=$(wc -l < "$OUT/$tag/games.jsonl" 2>/dev/null || echo 0)
  [ "$n" -gt 0 ] || alert "$tag: 0 games"
  grep -h "ALERT\|games, A wr\|Traceback\|Error" "$OUT/$tag.txt" | tee -a "$OUT/PROGRESS"
done
kill "$UP" 2>/dev/null
echo done > "$OUT/DONE"
