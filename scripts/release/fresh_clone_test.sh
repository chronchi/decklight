#!/usr/bin/env bash
# Release step 4: follow the public README on a clean box, from the generated public export only.
# Runs on vast (scripts/launch/launch-fresh-clone.sh ships data/cache/release-test/ with
# decklight.bundle = `git bundle` of the export, and the model tarball).
# It never uses the private copy pushed to /workspace/ygo-zero except to read those two files and to
# write runs/<name>/ (summary.txt: one "PASS|FAIL step" line each; ALERT on failure).
# Usage: bash scripts/release/fresh_clone_test.sh NAME
set -uo pipefail
NAME="${1:?usage: fresh_clone_test.sh NAME}"
PRIV="$(cd "$(dirname "$0")/../.." && pwd)"
IN="$PRIV/data/cache/release-test"
OUT="$PRIV/runs/$NAME"
WORK=/root/fresh
mkdir -p "$OUT/logs" "$WORK"
export PATH="$HOME/.local/bin:$PATH"
NP=$(nproc)
echo "cpus $NP, $(free -g | awk '/Mem/{print $2}') GiB RAM" | tee "$OUT/box.txt"

step() {  # step LABEL CMD...: run in the clone, log, record PASS/FAIL; returns the command's status
  local label="$1"; shift
  local t0=$SECONDS
  (cd "$WORK/decklight" 2>/dev/null || cd "$WORK"; "$@") > "$OUT/logs/$label.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "PASS $label ($((SECONDS - t0)) s)" | tee -a "$OUT/summary.txt"
  else
    echo "FAIL $label exit $rc ($((SECONDS - t0)) s)" | tee -a "$OUT/summary.txt"
    echo "ALERT fresh-clone: $label failed (exit $rc), see logs/$label.log" >> "$OUT/ALERT"
    tail -20 "$OUT/logs/$label.log"
  fi
  return $rc
}

# The README's steps, verbatim except: clone from the bundle (not yet on GitHub) and fetch the model
# bundle with --file (the GitHub release does not exist yet).
step install_uv bash -c 'command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh' || exit 1
step clone git clone --recursive "$IN/decklight.bundle" "$WORK/decklight" || exit 1
step submodules_pinned bash -c 'git submodule status | tee /dev/stderr | grep -qv "^[-+U]"' || true
step uv_sync uv sync --group dev --group ml --group web || exit 1
step build_ext scripts/build_ext.sh || exit 1
step fetch_release scripts/fetch_release.sh v0.1 --file "$IN/decklight-models-v0.1.tar.gz" || exit 1
step no_private_files bash -c '! ls NOTEBOOK.md STATUS.md AGENTS.md CLAUDE.md docs/briefs 2>/dev/null | grep .' || true

# Unit tests: the README's `uv run pytest tests/unit -q`, one file per core so it fits the cap.
run_units() {
  ls tests/unit/test_*.py | xargs -P "$NP" -I{} sh -c \
    'timeout 2400 uv run --no-sync pytest "$1" -q -p no:cacheprovider > "$2/logs/unit-$(basename "$1" .py).log" 2>&1; echo "$? $1"' _ {} "$OUT" \
    | sort -k2 > "$OUT/unit_rc.txt"
  echo "files $(wc -l < "$OUT/unit_rc.txt"), failed $(grep -vc '^0 ' "$OUT/unit_rc.txt")"
  grep -v '^0 ' "$OUT/unit_rc.txt" && return 1
  grep -h -E '[0-9]+ (passed|failed)' "$OUT"/logs/unit-*.log | awk '{for(i=1;i<NF;i++) if($(i+1)~/^passed/) p+=$i; else if($(i+1)~/^failed/) f+=$i} END {print "passed", p, "failed", f+0}'
}
step unit_tests run_units

# Benchmark path: gauntlet group + ygo-agent's tflite + one searched game of the product vs ygo-agent.
step gauntlet_sync uv sync --group dev --group ml --group web --group gauntlet
step tflite bash -c 'curl -fL --create-dirs -o data/models/0546_26550M.tflite https://github.com/sbl1996/ygo-agent/releases/download/v0.1/0546_26550M.tflite && echo "d4ad2a78b6a2a2c1a4560f8f12da6f4b3abb24735657115969e96d71a2364b10  data/models/0546_26550M.tflite" | sha256sum -c'
bench() {
  . scripts/product.sh
  timeout 2400 uv run python scripts/search_match.py --ckpt "$PRODUCT_CKPT" --value "$PRODUCT_VALUE" $PRODUCT_SEARCH \
    --opp ygoagent --ga-seeds 6 --deal-seed0 9000 --games-limit 1 --out runs/my-card || return $?
  n=$(cat runs/my-card/*/games-*.jsonl runs/my-card/games-*.jsonl 2>/dev/null | wc -l)
  echo "games written: $n"; [ "$n" -ge 1 ]
}
step benchmark_1game bench

# Web client: starts, serves the page with its token, then is stopped (by PID).
web() {
  uv run python scripts/web_play.py --port 8765 > "$OUT/logs/web_server.log" 2>&1 &
  local pid=$!
  local url=""
  for _ in $(seq 120); do
    url=$(grep -oE 'http://127\.0\.0\.1:8765/\?t=[A-Za-z0-9_-]+' "$OUT/logs/web_server.log" | head -1)
    [ -n "$url" ] && break; sleep 1
  done
  local code=0
  if [ -n "$url" ]; then
    sleep 3
    code=$(curl -s -o /dev/null -w '%{http_code}' "$url")
    echo "url ok, GET -> $code"
  else echo "no url printed in 120 s"
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  [ "$code" = 200 ]
}
step web_client web

grep -q '^FAIL' "$OUT/summary.txt" && { echo "fresh-clone: some steps failed"; exit 1; }
echo "fresh-clone: all steps passed"
