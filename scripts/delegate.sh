#!/usr/bin/env bash
# Run an OpenCode agent on a brief in the background, with a swappable model, and log the run for benchmarking.
#   scripts/delegate.sh <brief.md> [model] [tag]
#   model: full opencode id (default $YGO_AGENT_MODEL or opencode-go/muse-spark-1.3-contributor)
#   tag:   run name (default: brief basename + model short name)
# Writes data/cache/briefs/<tag>.{log,pid}; on exit appends a row to data/cache/briefs/runs.tsv:
#   tag  model  brief  start  end  seconds  exit  status(from the "## RESULT" block)
set -euo pipefail
cd "$(dirname "$0")/.."
brief=$1
model=${2:-${YGO_AGENT_MODEL:-opencode-go/muse-spark-1.3-contributor}}
short=${model##*/}
tag=${3:-$(basename "$brief" .md).$short}
dir=data/cache/briefs
runs=$dir/runs.tsv
[ -f "$runs" ] || printf 'tag\tmodel\tbrief\tstart\tend\tseconds\texit\tstatus\n' > "$runs"
(
  t0=$(date +%s)
  set +e
  taskset -c 0-3 opencode run --auto -m "$model" --dir . "Do the task in the attached brief." -f "$brief" > "$dir/$tag.log" 2>&1
  rc=$?
  t1=$(date +%s)
  status=$(grep -A2 '^## RESULT' "$dir/$tag.log" | grep -o 'status: *[a-z]*' | tail -1 | sed 's/status: *//')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$model" "$brief" "$(date -d @"$t0" +%FT%T)" \
    "$(date -d @"$t1" +%FT%T)" "$((t1 - t0))" "$rc" "${status:-none}" >> "$runs"
) &
echo $! > "$dir/$tag.pid"
echo "$tag pid $(cat "$dir/$tag.pid") model $model log $dir/$tag.log"
