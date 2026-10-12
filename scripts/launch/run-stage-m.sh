#!/usr/bin/env bash
# Retrain stage M (docs/retrain/plan-v3.md §Stage M; user 2026-10-11 "I want to be comprehensive from the beginning,
# going first, going second, mid-game, end-game"): muse plans our whole turn from sampled sw-v self-play states of every
# pool train deck (4 states per bucket per deck), verified strictly. Laptop, background; LLM calls are network-bound,
# so 8 workers; engine work stays on cores 0-3. Chunks of 10 decks so results land as it goes (plan_turns writes at the
# end of a run and skips states whose line already exists, so a rerun resumes).
# Pilot 2026-10-12 (49 states, budget 4000): 20 verified, 15 budget failures -> budget 8000 here.
# Usage: nohup setsid bash scripts/launch/run-stage-m.sh > data/cache/stage_m/sweep.log 2>&1 &
cd "$(dirname "$0")/../.."
OUT=${OUT:-data/demos/turns}
DECKS=$(uv run --no-sync python -c "
from pathlib import Path
from ygo_zero.model.bc_data import eval_only_decks
h, n = eval_only_decks()
print(' '.join(sorted(p.stem for p in Path('decks/pool').glob('*.ydk') if p.stem not in h and p.stem not in n)))")
set -- $DECKS
echo "$(date +%T) stage M sweep: $# decks -> $OUT"
while [ $# -gt 0 ]; do
  chunk=("${@:1:10}"); shift $(( $# < 10 ? $# : 10 ))
  echo "$(date +%T) chunk: ${chunk[*]}"
  taskset -c 0-3 uv run --no-sync python scripts/plan_turns.py --games 'data/demos/sw-v/*.jsonl' --per-bucket 4 \
    --decks "${chunk[@]}" --workers 8 --budget 8000 --out-dir "$OUT" 2>&1 | grep -v "falling back to cli" | tail -4
  echo "$(date +%T) verified so far: $(cat "$OUT"/*/*.jsonl 2>/dev/null | wc -l)"
done
echo "$(date +%T) DONE"
