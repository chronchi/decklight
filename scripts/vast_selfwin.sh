#!/usr/bin/env bash
# ADR 0007 data round on a many-core GPU box, generation only (distill is a separate job):
# 1. league self-play from BASE (scripts/gen_selfplay.py): new states on all pool decks + outcomes
# 2. in parallel on spare cores: more ygo-agent teacher games (scripts/gen_teacher.py)
# 3. multi-score window search over the self-play games (scripts/gen_window.py --value): every candidate gets
#    board / value / done (won within h turns) / cards at horizons 1 and 3; targets select value@h1, and the search
#    log keeps every score so other labellings are derived offline (scripts/relabel_window.py)
# Env: BASE (policy best.pt, required), VALUE (default data/cache/value_v2/best.pt), SP_GAMES (4000), SP_PROCS (16),
#   WIN_SEARCH (window search flags, default '--horizons 1 3 --select value@h1 --own-frac 0.1'),
#   TEACHER_GAMES (10000; 0 skips the stage), TEACHER_PROCS (cores/4), WINDOW_GAMES (1500), WPROCS (12), SAMPLES (2),
#   WCAP (window stage timeout, 120m).
# Usage: bash scripts/vast_selfwin.sh <name>. Results: runs/<name>/{selfplay,teacher,window}/p<k>/, selfwin.txt, ALERT
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
crashwatch() {
  declare -A seen=()
  while true; do
    for f in "$OUT"/selfplay-*.txt "$OUT"/teacher-*.txt "$OUT"/window-*.txt; do
      [ -e "$f" ] || continue
      base=$(basename "$f")
      [ -n "${seen[$base]:-}" ] && continue
      grep -q "^Traceback" "$f" 2>/dev/null || continue
      last=$(grep -v '^[[:space:]]*$' "$f" | tail -n 1 | cut -c1-200)
      echo "ALERT worker crashed: $base: $last" >> "$OUT/ALERT"
      seen[$base]=1
    done
    sleep "${CRASHWATCH_S:-60}"
  done
}
if [ "${1:-}" = "--crashwatch-only" ]; then OUT="${2:?usage: $0 --crashwatch-only <outdir>}"; crashwatch; exit 0; fi
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
: "${BASE:?set BASE to the policy best.pt}"
VALUE="${VALUE:-data/cache/value_v2/best.pt}"
SPG=${SP_GAMES:-4000}; SPP=${SP_PROCS:-16}; TG=${TEACHER_GAMES:-10000}
TP=${TEACHER_PROCS:-$(( $(nproc) / 4 ))}; [ "$TP" -lt 1 ] && TP=1
WG=${WINDOW_GAMES:-1500}; WP=${WPROCS:-12}; S=${SAMPLES:-2}; WCAP=${WCAP:-120m}
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
# fragmentation: per-proc GPU memory grew 0.45 -> 1.45 GB over sw3/sw4 and OOMed workers; search_match (S0b, 30 procs) has this
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
for f in "$BASE" "$VALUE"; do [ -f "$f" ] || { alert "missing $f"; exit 3; }; done
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg)"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
crashwatch &
WATCH_PID=$!
echo "selfplay $SPG/$SPP procs, teacher $TG/$TP procs, window $WG/$WP procs samples $S; base $BASE value $VALUE; cores $(nproc)" > "$OUT/selfwin.txt"
T0=$SECONDS
tick() { echo "$1 $((SECONDS - T0)) s" >> "$OUT/selfwin.txt"; }

# 2. teacher games in the background (CPU only)
TPER=$(( (TG + TP - 1) / TP ))
T_PIDS=()
[ "$TG" -gt 0 ] && for p in $(seq 0 $((TP - 1))); do
  timeout 150m uv run --no-sync python scripts/gen_teacher.py --out "$OUT/teacher/p$p" --games "$TPER" \
    --seed $(( 7000000 + p * 100000 )) --threads 2 --sample-frac 0.5 > "$OUT/teacher-$p.txt" 2>&1 &
  T_PIDS+=($!)
done

# 1. self-play
SPER=$(( (SPG + SPP - 1) / SPP ))
SP_PIDS=()
for p in $(seq 0 $((SPP - 1))); do
  timeout 50m uv run --no-sync python scripts/gen_selfplay.py --ckpt "$BASE" --device cuda --games "$SPER" --temp 1.0 \
    --league 'self:0.4,ygoagent:0.3,greedy:0.15,random:0.15' --seed $(( 3000000 + 100000 * p )) --threads 2 \
    --out "$OUT/selfplay/p$p" > "$OUT/selfplay-$p.txt" 2>&1 &
  SP_PIDS+=($!)
done
wait "${SP_PIDS[@]}"
tick selfplay
n_sp=$(cat "$OUT"/selfplay/p*/*.jsonl 2>/dev/null | wc -l)
echo "selfplay records $n_sp" >> "$OUT/selfwin.txt"
# fail loud: under half the games written means the generator is broken; the window stage would label nothing useful
[ "$n_sp" -ge $(( SPG / 2 )) ] || { alert "selfplay wrote $n_sp of $SPG games"; }

# 3. multi-score window search over the self-play games
# never a bare `wait`: it also waits for the usage logger, which never exits (sw-20261005-2323 hung 20 min)
if [ "$n_sp" -gt 0 ]; then
  W_PIDS=()
  for p in $(seq 0 $((WP - 1))); do
    timeout "$WCAP" uv run --no-sync python scripts/gen_window.py --ckpt "$BASE" --value "$VALUE" --device cuda --width 32 \
      --records "$OUT/selfplay/p*/*.jsonl" --games "$WG" --proc "$p" --nproc "$WP" --k 4 \
      --samples "$S" --rollout-temp 1.0 ${WIN_SEARCH:---horizons 1 3 --select value@h1 --own-frac 0.1} --seed 11 \
      --out "$OUT/window/p$p" > "$OUT/window-$p.txt" 2>&1 &
    W_PIDS+=($!)
  done
  wait "${W_PIDS[@]}"
  tick window
  grep -h "^done\|Traceback" "$OUT"/window-*.txt >> "$OUT/selfwin.txt"
  n_w=$(cat "$OUT"/window/p*/*.jsonl 2>/dev/null | wc -l)
  echo "window records $n_w" >> "$OUT/selfwin.txt"
  n_done=$(grep -l "^done" "$OUT"/window-*.txt | wc -l)
  [ "$n_done" -eq "$WP" ] || alert "window: $n_done of $WP procs finished (cap or crash)"
  [ "$n_w" -ge $(( WG / 2 )) ] || alert "window wrote $n_w of $WG games"
fi
[ "${#T_PIDS[@]}" -gt 0 ] && wait "${T_PIDS[@]}"
tick teacher
n_t=$(cat "$OUT"/teacher/p*/*.jsonl 2>/dev/null | wc -l)
echo "teacher records $n_t" >> "$OUT/selfwin.txt"
[ "$TG" -eq 0 ] || [ "$n_t" -ge $(( TG / 2 )) ] || alert "teacher wrote $n_t of $TG games"
du -sh "$OUT"/selfplay "$OUT"/window "$OUT"/teacher >> "$OUT/selfwin.txt" 2>&1
kill "$USAGE_PID" 2>/dev/null
kill "$WATCH_PID" 2>/dev/null
echo done > "$OUT/DONE"
