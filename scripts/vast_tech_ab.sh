#!/usr/bin/env bash
# Tech-card screen N4 (docs/plans/2026-10-16-engine-eval-and-metrics.md §3c, questions
# docs/techcards/questions-2026-10-16.json): 25 questions (5 decks x pos/neg/2 real/Called),
# pilot = opponent = data/cache/P/best.pt raw (no search), each question >= 2,000 pairs
# (NOPP opp decks x SEEDS seeds x 2 seats, default 10 x 100 x 2).
#
# Cost [M/E]: 25 questions x 2,000 pairs x 2 games + 4,000 A/A games = 104,000 games.
# Measured: ~8 games/s on a vast RL rollout (NOTEBOOK 2026-10-16; includes training overhead,
# so raw play should be faster) -> ~3.6 h worst case. Estimated raw-P play ~20 games/s [E] ->
# ~1.4 h ~= $X at $X/h. The probe below measures per-proc games/s before the waves:
# if 104,000 / measured_games_per_s > 6,500 s, stop and rerun with fewer questions (--only list).
# Env: Q (questions file), PILOT/OPP (default data/cache/P/best.pt), NOPP (10), SEEDS (100),
# SEED (opp-sampling seed, 0), SEED_BLOCK (seeds per unit, 25), VAR_CAP (per-unit cap, 50m),
# PROBE_SEEDS (5). Units are (question x seed block) so no core idles at the tail.
# First milestone: full-scale A/A (must be exactly 0, else the job stops) + first unit's
# numbers to PROGRESS. Usage: bash scripts/vast_tech_ab.sh <name>. Do NOT run on the laptop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_tech_ab.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT/units"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export HF_HOME="$PWD/data/cache/hf"
Q="${Q:-docs/techcards/questions-2026-10-16.json}"
PILOT="${PILOT:-data/cache/P/best.pt}"; OPP="${OPP:-$PILOT}"
NOPP="${NOPP:-10}"; SEEDS="${SEEDS:-100}"; SEED="${SEED:-0}"; BLOCK="${SEED_BLOCK:-25}"
[ -s "$PILOT" ] && [ -f "$Q" ] || { alert "missing pilot $PILOT or questions $Q"; exit 3; }
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
. scripts/cores.sh
J=$(box_cores)
# probe: one small unit to completion -> per-proc games/s, GPU mem, RSS; size the pool by measurement
Q0=$(uv run --no-sync python -c "import json; print(json.load(open('$Q'))[0]['id'])")
T0=$SECONDS
uv run --no-sync python scripts/tech_ab.py --questions "$Q" --only "$Q0" --skip-aa \
  --pilot "$PILOT" --opp "$OPP" --opp-decks decks/gauntlet --n-opp-decks "$NOPP" --seed "$SEED" \
  --seed0 0 --seeds "${PROBE_SEEDS:-5}" --threads 1 --num-envs 16 --device cuda \
  --out "$OUT/probe" > "$OUT/probe.txt" 2>&1 || { alert "probe failed: $(tail -1 "$OUT/probe.txt")"; exit 3; }
PSEC=$((SECONDS - T0))
GAMES=$(uv run --no-sync python -c "import json; print(2*json.load(open('$OUT/probe/$Q0/report.json'))['n_pairs'])")
GPS=$(python3 -c "print(${GAMES}/${PSEC})")
GMB=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
GTOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RTOT=$(free -m | awk '/Mem/{print $2}')
RMB=$(ps -o rss= -p $$ 2>/dev/null | awk '{print int($1/1024)}')
GMB=${GMB:-1500}; RMB=${RMB:-1500}
NP=$(python3 -c "print(max(1, min($J, int($GTOT*0.85/($GMB*1.5)), int($RTOT*0.8/($RMB*1.3)))))")
ENVS=$(python3 -c "print(max(4, min(32, $J*2//max(1,$NP))))")
echo "cores $J, gpu ${GTOT}MB, ram ${RTOT}MB; probe ${GPS} games/s gpu ${GMB}MB rss ${RMB}MB -> $NP procs x $ENVS envs" \
  | tee "$OUT/sizing.txt"
TOTAL=$((25 * NOPP * SEEDS * 2 * 2 + NOPP * SEEDS * 2 * 2))
ETA=$(python3 -c "print($TOTAL/($GPS*$NP)/3600)")
echo "$(date +%T) milestone: probe ${GPS} games/s/proc x $NP procs; ETA ${ETA} h for $TOTAL games" >> "$OUT/PROGRESS"
if python3 -c "exit(0 if $TOTAL/($GPS*$NP) > 6500 else 1)"; then
  alert "ETA ${ETA} h exceeds the 2 h cap at measured throughput; rerun with fewer questions"
  exit 5
fi
# milestone 1: full-scale A/A must be exactly 0, else stop before spending on questions
AADECK=$(uv run --no-sync python -c "import json; print(json.load(open('$Q'))[0]['deck'])")
T0=$SECONDS
uv run --no-sync python scripts/tech_ab.py --aa-only "$AADECK" \
  --pilot "$PILOT" --opp "$OPP" --opp-decks decks/gauntlet --n-opp-decks "$NOPP" --seed "$SEED" \
  --seed0 0 --seeds "$SEEDS" --threads 1 --num-envs "$ENVS" --device cuda \
  --out "$OUT/aa" > "$OUT/aa.txt" 2>&1 || { alert "A/A failed: $(tail -1 "$OUT/aa.txt")"; exit 4; }
echo "$(date +%T) milestone: A/A exactly 0 at full scale ($((NOPP * SEEDS * 2)) pairs, $((SECONDS - T0)) s)" >> "$OUT/PROGRESS"
# waves of (question x seed block) units; each unit self-contains its rows for --collect
IDS=$(uv run --no-sync python -c "import json; print('\n'.join(q['id'] for q in json.load(open('$Q'))))")
UNITS=()
for id in $IDS; do
  for ((s = 0; s < SEEDS; s += BLOCK)); do
    b=$(( s + BLOCK > SEEDS ? SEEDS - s : BLOCK ))
    UNITS+=("$id,$s,$b")
  done
done
echo "${UNITS[@]}" | tr ' ' '\n' > "$OUT/units.txt"
run_unit() {
  local spec="$1" id s b u
  id="${spec%%,*}"; s="${spec#*,}"; s="${s%%,*}"; b="${spec##*,}"; u="$OUT/units/u-$id-$s"
  [ -f "$u/DONE" ] && return 0
  mkdir -p "$u"
  if timeout "${VAR_CAP:-50m}" uv run --no-sync python scripts/tech_ab.py --questions "$Q" --only "$id" --skip-aa \
      --pilot "$PILOT" --opp "$OPP" --opp-decks decks/gauntlet --n-opp-decks "$NOPP" --seed "$SEED" \
      --seed0 "$s" --seeds "$b" --threads 1 --num-envs "$ENVS" --device cuda \
      --out "$u" > "$u/proc.txt" 2>&1; then
    echo "$(date +%T) unit $id seed0 $s: $(grep -h . "$u/$id/summary.md" 2>/dev/null | sed -n 4p)" >> "$OUT/PROGRESS"
    echo done > "$u/DONE"
  else
    echo "ALERT unit $id seed0 $s exit $?: $(tail -1 "$u/proc.txt")" | tee -a "$OUT/ALERT"
    return 1
  fi
}
export -f run_unit alert
export OUT Q PILOT OPP NOPP SEED BLOCK ENVS VAR_CAP
echo "${UNITS[@]}" | tr ' ' '\n' | xargs -P "$NP" -I{} bash -c 'run_unit "$@"' _ {} || alert "some units failed (see above)"
grep -h "^ALERT\|Traceback" "$OUT"/units/u-*/proc.txt >> "$OUT/ALERT" 2>/dev/null || true
# first question's numbers to PROGRESS as the milestone read
FIRST_U=$(ls -d "$OUT"/units/u-"$Q0"-* 2>/dev/null | sort -t- -k3 -n | head -1)
[ -n "${FIRST_U:-}" ] && sed "s/^/$(date +%T) first-question /" "$FIRST_U"/*/summary.md 2>/dev/null | head -12 >> "$OUT/PROGRESS"
# collect units + full-scale A/A into the final per-question reports and overall table
uv run --no-sync python scripts/tech_ab.py --collect "$OUT"/units/u-* "$OUT/aa" --out "$OUT/card" \
  > "$OUT/collect.txt" 2>&1 || { alert "collect failed: $(tail -1 "$OUT/collect.txt")"; exit 4; }
cat "$OUT/card/overall.md" >> "$OUT/PROGRESS"
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
