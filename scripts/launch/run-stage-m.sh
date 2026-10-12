#!/usr/bin/env bash
# Retrain stage M (docs/retrain/plan-v3.md §Stage M; user 2026-10-11 "I want to be comprehensive from the beginning,
# going first, going second, mid-game, end-game"): muse plans our whole turn from sampled sw-v self-play states of every
# pool train deck (4 states per bucket per deck), verified strictly. Laptop, background; LLM calls are network-bound,
# so 8 workers; engine work stays on cores 0-3. Chunks of 10 decks so results land as it goes (plan_turns writes at the
# end of a run and skips states whose line already exists, so a rerun resumes).
# Pilot 2026-10-12 (49 states, budget 4000): 20 verified, 15 budget failures -> budget 8000 here.
# plan_turns makes all LLM calls of a chunk, then verifies on one core, so one stream leaves cores idle (2026-10-12: a
# chunk took ~2 h). STREAM/NSTREAMS split the chunks round-robin over parallel streams (all on cores 0-3); FROM_CHUNK
# skips chunks already handled by an earlier sweep.
# LLM path: the HTTP endpoint answers "Model is unavailable" for muse (2026-10-12), so calls fall back to the CLI, which
# holds 2 machine-wide slots: ~2 calls/min and "no free slot in 900 s" errors (30 of 115 chunk-0 states). Start
# `sh scripts/llm_server.sh start` and set YGO_OPENCODE_SERVER=http://127.0.0.1:4097 (no slot cap, ~7 s per call).
# A final pass over all decks (FROM_CHUNK=0, NSTREAMS=1) retries the states that failed on llm errors.
# Usage: nohup setsid bash scripts/launch/run-stage-m.sh > data/cache/stage_m/sweep.log 2>&1 &
#        for i in 0 1 2; do STREAM=$i NSTREAMS=3 FROM_CHUNK=1 nohup setsid bash scripts/launch/run-stage-m.sh \
#          > data/cache/stage_m/sweep-$i.log 2>&1 & done
cd "$(dirname "$0")/../.."
OUT=${OUT:-data/demos/turns}
STREAM=${STREAM:-0}; NSTREAMS=${NSTREAMS:-1}; FROM_CHUNK=${FROM_CHUNK:-0}; WORKERS=${WORKERS:-8}
DECKS=$(uv run --no-sync python -c "
from pathlib import Path
from ygo_zero.model.bc_data import eval_only_decks
h, n = eval_only_decks()
print(' '.join(sorted(p.stem for p in Path('decks/pool').glob('*.ydk') if p.stem not in h and p.stem not in n)))")
set -- $DECKS
echo "$(date +%T) stage M sweep: $# decks -> $OUT (stream $STREAM/$NSTREAMS from chunk $FROM_CHUNK)"
c=-1
while [ $# -gt 0 ]; do
  chunk=("${@:1:10}"); shift $(( $# < 10 ? $# : 10 )); c=$((c + 1))
  if [ $c -lt "$FROM_CHUNK" ] || [ $((c % NSTREAMS)) -ne "$STREAM" ]; then continue; fi
  echo "$(date +%T) chunk: ${chunk[*]}"
  taskset -c 0-3 uv run --no-sync python scripts/plan_turns.py --games 'data/demos/sw-v/*.jsonl' --per-bucket 4 \
    --decks "${chunk[@]}" --workers "$WORKERS" --budget 8000 --out-dir "$OUT" 2>&1 | grep -v "falling back to cli" | tail -4
  echo "$(date +%T) verified so far: $(cat "$OUT"/*/*.jsonl 2>/dev/null | wc -l)"
done
echo "$(date +%T) DONE"
