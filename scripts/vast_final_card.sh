#!/usr/bin/env bash
# Final report card, PLAN §0a P3–P4 (user 2026-10-10: "Let's go with 1 for now"; decisions in NOTEBOOK 2026-10-10 15:40):
# base_r3 + hppo search (honest, k4 s4 h1, value_ppo3, conf-skip 0.9) on FRESH deals (duel seeds from 1000, never used
# to pick the config), paired with base_r3 raw on the same deals; guards: search vs raw base_r3 on the holdout pool
# decks and on the unseen-archetype decks (decks/newcards); then single-game latency (one process, one game in flight).
# Env: GA_SEEDS (6 = 1,488 ygo-agent deals), HOLD_SEEDS (2 = 1,088 games), NEW_SEEDS (18 = 1,080 games), VAR_CAP.
# Usage: bash scripts/vast_final_card.sh <name>. Do NOT run on the laptop (vast GPU only).
# Results: runs/<name>/{raw,hppo,hold,new}/games-*.jsonl + summary.json, runs/<name>/latency/.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
H="--honest 1 --det-tries 32 --det-fallback skip --value data/cache/value_ppo3/best.pt --horizon 1 --claim 1"
V="raw|--opp ygoagent --raw 1 --ga-seeds ${GA_SEEDS:-6}"
V="$V;hppo|--opp ygoagent --ga-seeds ${GA_SEEDS:-6} $H"
V="$V;hold|--opp self --deck-set holdout --seeds ${HOLD_SEEDS:-2} $H"
V="$V;new|--opp self --deck-set newcards --seeds ${NEW_SEEDS:-18} $H"
CKPT=data/cache/base_r3/best.pt SM_ARGS="--k 4 --samples 4 --conf-skip 0.9 --deal-seed0 1000" VARIANTS="$V" \
  PROBE_ARGS="--opp ygoagent --ga-seeds ${GA_SEEDS:-6} $H" PROBE_NPROC=40 VAR_CAP="${VAR_CAP:-180m}" \
  bash scripts/vast_search_match.sh "$NAME"
# single-game latency: one process alone on the box, one game at a time, the first 6 fresh ygo-agent deals
L="$OUT/latency"; mkdir -p "$L"
T0=$(date +%s.%N)
# shellcheck disable=SC2086
uv run --no-sync python scripts/search_match.py --ckpt data/cache/base_r3/best.pt --device cuda --out "$L" --nproc 1 \
  --proc 0 --k 4 --samples 4 --conf-skip 0.9 --deal-seed0 1000 --opp ygoagent --ga-seeds 1 --games-limit 6 \
  --honest 1 --det-tries 32 --det-fallback skip --value data/cache/value_ppo3/best.pt --horizon 1 --claim 0 --width 1 \
  > "$L/proc-0.txt" 2>&1 || alert "latency run failed: $(tail -1 "$L/proc-0.txt")"
T1=$(date +%s.%N)
uv run --no-sync python scripts/search_match.py --summarize "$L" > "$L/summary.txt" 2>&1
python3 -c "print(f'latency: 6 games in {$T1 - $T0:.0f} s wall, {($T1 - $T0) / 6:.0f} s/game incl. model load')" \
  | tee -a "$OUT/PROGRESS"
