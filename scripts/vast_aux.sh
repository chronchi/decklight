#!/usr/bin/env bash
# Aux evaluator heads N3 (docs/plans/2026-10-16-engine-eval-and-metrics.md N3): P-vs-P games,
# aux-target shards, then train_aux.py on the frozen value_p2 trunk.
#   a. games: gen_selfplay.py --ckpt data/cache/P/best.pt --league self:1.0 --temp 1.0
#      --decks decks/pool (P vs P, no ygo-agent), GAMES games split over measured procs;
#   b. shards: build_aux_shards.py (CPU, all cores);
#   c. train: train_aux.py --value data/cache/value_p2/best.pt (GPU).
#   outputs in runs/<name>/: games/p*/, shards/, aux/{aux.pt, metrics.json}.
# Cost [E]: ~20,000 games at the probe-measured games/s + shards + a short GPU train;
# expected ~$X on a small GPU box (RTX 3060-class, >= 16 cores, <= $X/h, 2 h cap).
# Env: GAMES (20000), SEED (5000000), THREADS (8), PROBE_GAMES (60), GAMES_CAP (per-proc
# cap, 50m), STRIDE (4), EPOCHS (10). Units are per-proc seed blocks so no core idles.
# Milestones to PROGRESS: games/s after the probe, shard sample count, first-epoch
# holdout R2 per target. Usage: bash scripts/vast_aux.sh <name>. Do NOT run on the laptop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_aux.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT/games"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export HF_HOME="$PWD/data/cache/hf"
P="${P:-data/cache/P/best.pt}"; VAL="${VAL:-data/cache/value_p2/best.pt}"
GAMES="${GAMES:-20000}"; SEED="${SEED:-5000000}"; THREADS="${THREADS:-8}"
PROBE_GAMES="${PROBE_GAMES:-60}"; STRIDE="${STRIDE:-4}"; EPOCHS="${EPOCHS:-10}"
[ -s "$P" ] && [ -s "$VAL" ] || { alert "missing pilot $P or value $VAL"; exit 3; }
uv sync --group dev --group ml -q --no-install-package torch
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
# probe: one proc plays PROBE_GAMES games (~1-3 min); per-proc GPU memory and RSS size the pool.
# NOTEBOOK 2026-10-16 17:45: few processes with many envs/threads beat many small ones on a small GPU.
T0=$SECONDS
uv run --no-sync python scripts/gen_selfplay.py --ckpt "$P" --league 'self:1.0' --temp 1.0 \
  --decks decks/pool --seed "$SEED" --threads "$THREADS" --num-envs "${NUM_ENVS:-256}" --device cuda \
  --games "$PROBE_GAMES" --out "$OUT/probe" > "$OUT/probe.txt" 2>&1 &
PP=$!; GMB=0; RMB=0
while kill -0 $PP 2>/dev/null; do
  g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  r=$(ps -o rss= --ppid $PP 2>/dev/null | sort -n | tail -1)
  [ "${g:-0}" -gt "$GMB" ] && GMB=$g; [ $(( ${r:-0} / 1024 )) -gt "$RMB" ] && RMB=$(( ${r:-0} / 1024 ))
  sleep 2
done
wait $PP || { alert "probe failed: $(tail -1 "$OUT/probe.txt")"; exit 3; }
PSEC=$((SECONDS - T0))
GPS=$(python3 -c "print($PROBE_GAMES/max($PSEC,1))")
GTOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RTOT=$(free -m | awk '/Mem/{print $2}')
[ "$GMB" -gt 0 ] || { alert "probe: no GPU memory reading"; exit 3; }
[ "$RMB" -gt 0 ] || RMB=1500
NP=${NP:-$(python3 -c "print(max(1, min($J // $THREADS, int($GTOT*0.9/($GMB*1.2)), int($RTOT*0.8/($RMB*1.3)))))")}
echo "cores $J, gpu ${GTOT}MB, ram ${RTOT}MB; probe ${GPS} games/s gpu ${GMB}MB rss ${RMB}MB -> $NP procs x $THREADS threads" \
  | tee "$OUT/sizing.txt"
echo "$(date +%T) milestone: probe ${GPS} games/s/proc x $NP procs" >> "$OUT/PROGRESS"
rm -rf "$OUT/probe"
# a. games in per-proc seed blocks
GP=$(( (GAMES + NP - 1) / NP ))
T0=$SECONDS
PIDS=()
for ((p = 0; p < NP; p++)); do
  timeout "${GAMES_CAP:-50m}" uv run --no-sync python scripts/gen_selfplay.py --ckpt "$P" --league 'self:1.0' \
    --temp 1.0 --decks decks/pool --seed $(( SEED + 100000 * (p + 1) )) --threads "$THREADS" --num-envs "${NUM_ENVS:-256}" --device cuda \
    --games "$GP" --out "$OUT/games/p$p" > "$OUT/games-$p.txt" 2>&1 &
  PIDS+=($!)
done
rc=0
for pid in "${PIDS[@]}"; do wait "$pid" || { alert "games proc pid $pid exit $?"; rc=1; }; done
[ $rc -eq 0 ] || exit 4
grep -h "^ALERT\|Traceback" "$OUT"/games-*.txt >> "$OUT/ALERT" 2>/dev/null || true
GOT=$(cat "$OUT"/games/p*/*.jsonl 2>/dev/null | wc -l)
echo "$(date +%T) milestone: games $GOT/$GAMES in $((SECONDS - T0)) s" >> "$OUT/PROGRESS"
[ "$GOT" -gt 0 ] || { alert "0 games written"; exit 4; }
# b. shards on all CPU cores
T0=$SECONDS
uv run --no-sync python scripts/build_aux_shards.py --records "$OUT/games/p*/*.jsonl" \
  --out "$OUT/shards" --stride "$STRIDE" --jobs "$J" > "$OUT/shards.txt" 2>&1 \
  || { alert "shards failed: $(tail -1 "$OUT/shards.txt")"; exit 4; }
grep -h "^ALERT\|Traceback" "$OUT/shards.txt" >> "$OUT/ALERT" 2>/dev/null || true
NS=$(uv run --no-sync python -c "
import numpy as np, glob
n = m = 0
for f in glob.glob('$OUT/shards/*.npz'):
    p = np.load(f); n += len(p['game']); m += int(p['mask'].sum())
print(f'{n} {m}')")
echo "$(date +%T) milestone: shards $NS (samples has_next) in $((SECONDS - T0)) s" >> "$OUT/PROGRESS"
[ "${NS%% *}" -gt 0 ] 2>/dev/null || { alert "0 shard samples"; exit 4; }
# c. aux heads on the GPU
uv run --no-sync python scripts/train_aux.py --shards "$OUT/shards" --value "$VAL" \
  --out "$OUT/aux" --epochs "$EPOCHS" --device cuda > "$OUT/train.txt" 2>&1
TRC=$?
grep -h "^ALERT\|Traceback" "$OUT/train.txt" >> "$OUT/ALERT" 2>/dev/null || true
R2=$(grep '"holdout_r2"' "$OUT/train.txt" | head -1)
echo "$(date +%T) milestone: first-epoch $R2 (train exit $TRC)" >> "$OUT/PROGRESS"
[ -s "$OUT/aux/aux.pt" ] && [ -s "$OUT/aux/metrics.json" ] || { alert "missing aux.pt/metrics.json"; exit 4; }
[ $TRC -eq 0 ] || { alert "train_aux exit $TRC (see train.txt; metrics.json still written)"; exit $TRC; }
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
