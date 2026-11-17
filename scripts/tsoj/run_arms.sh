#!/usr/bin/env bash
# One tsoj-arms run: P (or a forced-activation arm) vs tsoj's alpha-beta bot.
#   scripts/tsoj/run_arms.sh TAG ARM NODES N CONC [BOOK]
# ARM=p|top@K|rand@K (OUTNAME=p2 with ARM=p: the determinism repeat); NODES=BOT_SEARCH_NODES (per decision); N openings; CONC games at once.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="${1:?TAG}"; ARM="${2:-p}"; NODES="${3:-2000}"; N="${4:-1}"; CONC="${5:-1}"
BOOK="${6:-$ROOT/scripts/tsoj/books/tsoj.jsonl}"
YGO_AI="${YGO_AI:-$ROOT/../yugioh_ai}"
T="$YGO_AI/ygopro-core-rust/target/release"
OUT="$ROOT/runs/tsoj-arms/$TAG"
mkdir -p "$OUT/logs" "$OUT/records"
[ -x "$T/ygo-arbiter" ] || { echo "missing $T/ygo-arbiter (run scripts/tsoj/build_tsoj.sh)" >&2; exit 2; }
[ -x "$T/bot" ] || { echo "missing $T/bot" >&2; exit 2; }
[ -x "$T/ygo-windbot" ] || { echo "missing $T/ygo-windbot" >&2; exit 2; }

export DL_TAG="$TAG" DL_ARM="$ARM" DL_ARM_SEED="${DL_ARM_SEED:-0}" \
       DL_ARMS_LOG="$OUT/arms.jsonl" PYTHONUNBUFFERED=1
export YGO_ROOT="$YGO_AI"
START=$(date +%s)
"$T/ygo-arbiter" \
  --bot "ab=env YGO_ROOT=$YGO_AI BOT_SEARCH_NODES=$NODES $T/bot alphabeta ${BOT_SEED:-1}" \
  --bot "dl=$T/ygo-windbot" \
  --option "dl.WindBot=$ROOT/scripts/tsoj/client" \
  --book "$BOOK" -n "$N" -c "$CONC" \
  --records "$OUT/${OUTNAME:-$ARM}.jsonl" > "$OUT/${OUTNAME:-$ARM}.log" 2>&1 || true
END=$(date +%s)
echo "run_arms: $TAG $ARM $((END-START))s -> $OUT/${OUTNAME:-$ARM}.jsonl"
uv run --project "$ROOT" python "$ROOT/scripts/tsoj/check_run.py" \
  --records "$OUT/${OUTNAME:-$ARM}.jsonl" --arms "$OUT/arms.jsonl" --arm "$ARM" --alert "$OUT/ALERT"
