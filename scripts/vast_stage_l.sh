#!/usr/bin/env bash
# Retrain stage L (docs/retrain/plan-v2.md §Stage L, plan-v3): honest decisive window labels (hw).
# 1. self-play by base_r3 vs itself (league self:1) on pool train decks only (gen_selfplay excludes decks/pool/HOLDOUT
#    and kewl-tune; newcards/gauntlet are other dirs), both seats, all turns;
# 2. decisive-label search over those games (scripts/gen_window.py --ho 1 --windows all): chain windows on either turn
#    + own Main-phase activation decisions (own-frac), merged candidates, honest worlds, value_ppo3 at h1, samples
#    4 -> 12 in steps of 4; a target only where the best candidate beats the runner-up at 2 SE and >= 0.05.
# Sizing: one gen_window probe (150 s peak GPU memory / RSS), procs = min(cores, GPU, RAM).
# First milestone: when the first proc has written 10 games, the decisive-label rate and the window fire rate are read
# (scripts/audit/stage_l_milestone.py); 0 labels or a window fire rate > 0.5 (sw-v's 0.60, ygo-agent 0.24) kills the job.
# Env: BASE, VALUE, SP_GAMES (3000), WG (window games, 1500), OWN_FRAC (0.3), WCAP (150m).
# Usage: bash scripts/vast_stage_l.sh <name>. Vast GPU only.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
BASE="${BASE:-data/cache/base_r3/best.pt}"; VALUE="${VALUE:-data/cache/value_ppo3/best.pt}"
SPG=${SP_GAMES:-3000}; WG=${WG:-1500}; OWN_FRAC=${OWN_FRAC:-0.3}; WCAP=${WCAP:-150m}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
for f in "$BASE" "$VALUE"; do [ -f "$f" ] || { alert "missing $f"; exit 3; }; done
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
J=$(nproc); [ "$Q" != max ] && J=$(( Q / P ))
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }

# 1. self-play (GPU policy, small per-proc memory): one proc per 2 cores, at most 24
SPP=$(( J / 2 )); [ "$SPP" -gt 24 ] && SPP=24; [ "$SPP" -lt 1 ] && SPP=1
SPER=$(( (SPG + SPP - 1) / SPP ))
SP_PIDS=()
for p in $(seq 0 $((SPP - 1))); do
  timeout 40m uv run --no-sync python scripts/gen_selfplay.py --ckpt "$BASE" --device cuda --games "$SPER" --temp 1.0 \
    --league 'self:1' --seed $(( 5000000 + 100000 * p )) --threads 2 --out "$OUT/selfplay/p$p" \
    > "$OUT/selfplay-$p.txt" 2>&1 &
  SP_PIDS+=($!)
done
for pid in "${SP_PIDS[@]}"; do wait "$pid" || alert "selfplay proc exit $?"; done
n_sp=$(cat "$OUT"/selfplay/p*/*.jsonl 2>/dev/null | wc -l)
tick "selfplay $n_sp games ($SPP procs)"
[ "$n_sp" -ge $(( SPG / 2 )) ] || { alert "selfplay wrote $n_sp of $SPG games"; exit 2; }
# eval-only decks must not appear (holdout / newcards / gauntlet stems)
uv run --no-sync python - "$OUT" <<'EOF' || { alert "eval-only deck in selfplay"; exit 2; }
import glob, json, sys
from ygo_zero.model.bc_data import eval_only_decks
hold, new = eval_only_decks()
bad = {d for f in glob.glob(f"{sys.argv[1]}/selfplay/p*/*.jsonl") for x in open(f) for d in json.loads(x)["decks"]
       if d in hold or d in new}
print("eval-only decks in selfplay:", sorted(bad)); sys.exit(1 if bad else 0)
EOF

WARGS="--ckpt $BASE --value $VALUE --device cuda --width 32 --k 4 --horizon 1 --samples 4 --rollout-temp 1.0
  --honest 1 --det-tries 32 --det-fallback skip --conf-skip 0.9 --ho 1 --ho-z ${HO_Z:-3} --ho-min-gap 0.05 --ho-s-max 12
  --ho-step 4 --windows all --own-frac $OWN_FRAC --seed 13"
# 2a. probe one gen_window proc for 150 s
# shellcheck disable=SC2086
uv run --no-sync python scripts/gen_window.py $WARGS --records "$OUT/selfplay/p*/*.jsonl" --games 40 \
  --out "$OUT/probe" > "$OUT/probe.txt" 2>&1 &
PP=$!; GMB=0; RMB=0
for i in $(seq 15); do
  sleep 10
  kill -0 $PP 2>/dev/null || { alert "probe died before 150 s: $(tail -1 "$OUT/probe.txt")"; exit 3; }
  g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -n | tail -1)
  r=$(ps -o rss= --ppid $PP 2>/dev/null | sort -n | tail -1)
  [ "${g:-0}" -gt "$GMB" ] && GMB=$g; [ $(( ${r:-0} / 1024 )) -gt "$RMB" ] && RMB=$(( ${r:-0} / 1024 ))
done
kill $PP 2>/dev/null; sleep 3; for c in $(ps -o pid= --ppid $PP 2>/dev/null); do kill "$c" 2>/dev/null; done
GTOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RTOT=$(free -m | awk '/Mem/{print $2}')
[ "$GMB" -gt 0 ] || { alert "probe: no GPU memory reading"; exit 3; }
[ "$RMB" -gt 0 ] || RMB=2000
NP=$(python3 -c "print(max(1, min($J, int($GTOT*0.85/($GMB*${GPU_MARGIN:-1.3})), int($RTOT*0.8/($RMB*1.3)))))")
echo "cores $J, gpu ${GTOT}MB, ram ${RTOT}MB; probe gpu ${GMB}MB rss ${RMB}MB per proc -> $NP procs" | tee "$OUT/sizing.txt"

# 2b. decisive-label search, small units per proc (games split by --proc/--nproc)
W_PIDS=()
for p in $(seq 0 $((NP - 1))); do
  # shellcheck disable=SC2086
  timeout "$WCAP" uv run --no-sync python scripts/gen_window.py $WARGS --records "$OUT/selfplay/p*/*.jsonl" \
    --games "$WG" --proc "$p" --nproc "$NP" --out "$OUT/hw/p$p" > "$OUT/hw-$p.txt" 2>&1 &
  W_PIDS+=($!)
done
( while sleep 60; do
    n=$(cat "$OUT"/hw/p*/*.jsonl 2>/dev/null | wc -l)
    if [ "$n" -ge $(( NP * 3 )) ]; then
      if uv run --no-sync python scripts/audit/stage_l_milestone.py "$OUT/hw" > "$OUT/milestone.txt" 2>&1; then
        echo "$(date +%T) $(tail -1 "$OUT/milestone.txt")" >> "$OUT/PROGRESS"
      else
        alert "$(tail -1 "$OUT/milestone.txt"); killing the search"
        touch "$OUT/KILLED"
        for pid in "${W_PIDS[@]}"; do kill "$pid" 2>/dev/null; done  # timeout forwards the TERM to gen_window
      fi
      break
    fi
  done ) &
for pid in "${W_PIDS[@]}"; do wait "$pid" || alert "hw proc exit $?"; done
[ -f "$OUT/KILLED" ] && exit 2
n_w=$(cat "$OUT"/hw/p*/*.jsonl 2>/dev/null | wc -l)
tick "hw $n_w games"
grep -h "^done" "$OUT"/hw-*.txt | head -3 >> "$OUT/PROGRESS"
[ "$n_w" -ge $(( WG / 2 )) ] || alert "hw wrote $n_w of $WG games"
uv run --no-sync python scripts/audit/stage_l_milestone.py "$OUT/hw" > "$OUT/final.txt" 2>&1 || alert "final: $(tail -1 "$OUT/final.txt")"
echo "final $(tail -1 "$OUT/final.txt")" >> "$OUT/PROGRESS"
# 3. replication audit of the decisive labels (plan v2: >= 0.85 must keep their sign with 32 fresh worlds)
PIDS=()
for ((p = 0; p < NP; p++)); do
  timeout 45m uv run --no-sync python scripts/audit/replicate_overrides.py --hw --games "$OUT/hw" --n 500 --samples 32 \
    --ckpt "$BASE" --value "$VALUE" --nproc "$NP" --proc "$p" --out "$OUT/repl/hw.jsonl" > "$OUT/repl-hw-$p.txt" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || alert "repl hw: proc exit $?"; done
if uv run --no-sync python scripts/audit/replicate_overrides.py --hw --games "$OUT/hw" --ckpt x --value x \
    --out "$OUT/repl/hw.jsonl" --summarize > "$OUT/repl-hw.txt" 2>&1; then
  echo "repl hw $(tail -1 "$OUT/repl-hw.txt")" >> "$OUT/PROGRESS"
  uv run --no-sync python -c "import json,sys; s=json.load(open('$OUT/repl/hw.summary.json')); sys.exit(s['sign_holds'] < 0.85)" \
    || alert "repl hw precision below 0.85: $(tail -1 "$OUT/repl-hw.txt") (plan: L labels are not used)"
else
  alert "repl hw summarize: $(tail -1 "$OUT/repl-hw.txt")"
fi
kill "$USAGE_PID" 2>/dev/null
echo done > "$OUT/DONE_ALL"
