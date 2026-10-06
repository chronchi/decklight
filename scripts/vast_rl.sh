#!/usr/bin/env bash
# RL v0 (PLAN Phase 5 first feel, 2026-10-05): PPO self-play from the BC checkpoint in data/cache/rl_init/
# (best.pt + metrics.jsonl), KL-anchored to it; gauntlet eval (vs BC, vs ygo-agent) at iteration 0, every
# EVAL_EVERY iterations and at the end. Usage: bash scripts/vast_rl.sh <name> [smoke|full]. Env: RL_HOURS (default 1.5),
# RL_ARGS (extra train_rl.py args), CARD (default 1; 0 skips the report-card stage), CARD_K (default 2 checkpoints
# for the card), CARD_PT_ARGS (playtest args for the card). Results in runs/<name>/: rl/metrics.jsonl + iter-*.pt,
# train.txt, PROGRESS (eval lines), card/ (chosen checkpoints + report card).
# The full run starts directly: its iteration-0 eval and train_rl's own fail-fast checks replace the old smoke
# stage. MODE=smoke runs only the tiny run (--iters 3 ...) for a quick crash check.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; MODE="${2:-full}"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
if [ "$Q" = "max" ]; then J=$(nproc); else J=$(( Q / P )); fi
[ "$J" -lt 2 ] && J=2
T0=$(date +%s)
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
echo "deps install $(( $(date +%s) - T0 ))s" >> "$OUT/torch.txt"
{ nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; echo "jobs $J"; free -g | head -2;
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; } > "$OUT/machine.txt" 2>&1
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA torch: abort" | tee -a "$OUT/torch.txt"; exit 3; }
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -m | awk '/Mem/{print $3}')"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
INIT=${INIT:-data/cache/rl_init/best.pt}
rl() {  # <out> <args...>
  local o="$1"; shift
  # shellcheck disable=SC2086
  uv run --no-sync python scripts/train_rl.py --init "$INIT" --out "$o" --device cuda "$@" ${RL_ARGS:-}
}
if [ "$MODE" = "smoke" ]; then
  rl "$OUT/smoke" --iters 3 --rollout 4096 --envs 64 --eval-every 2 --eval-seeds 1 --ga-seeds 1 --value-warmup 1 \
    > "$OUT/smoke.txt" 2>&1
  echo "exit $?" >> "$OUT/smoke.txt"
  rm -rf "$OUT/smoke"/*.pt
  kill "$USAGE_PID" 2>/dev/null
  echo done > "$OUT/DONE"
  exit 0
fi
# full run starts directly (no separate smoke stage): iteration-0 eval + fail-fast checks cover it
touch "$OUT/train.txt" "$OUT/PROGRESS"
# eval lines → PROGRESS (the launcher relays new lines); a plain loop so `kill $PROG_PID` stops it (no tail pipe)
( while sleep 30; do grep '^\[eval it' "$OUT/train.txt" > "$OUT/PROGRESS.tmp" 2>/dev/null && mv "$OUT/PROGRESS.tmp" "$OUT/PROGRESS"; done ) &
PROG_PID=$!
rl "$OUT/rl" --hours "${RL_HOURS:-1.5}" --eval-every "${EVAL_EVERY:-10}" > "$OUT/train.txt" 2>&1
TRAIN_RC=$?
echo "exit $TRAIN_RC" >> "$OUT/train.txt"
if [ "$TRAIN_RC" -ne 0 ]; then
  alert "train_rl exit $TRAIN_RC: skipping card stage"
fi
# keep the eval checkpoints small: last.pt (with optimizer) only for a resume; drop it to save the pull
rm -f "$OUT/rl/last.pt"
# card stage (same box): pick top CARD_K iters by eval.vs_bc.wr (excluding iter 0) and report-card them
if [ "$TRAIN_RC" -eq 0 ] && [ "${CARD:-1}" != "0" ]; then
  if [ ! -f "$OUT/rl/metrics.jsonl" ]; then
    alert "no $OUT/rl/metrics.jsonl: skipping card stage"
  else
    CHOSEN=$(python3 - "$OUT/rl/metrics.jsonl" "${CARD_K:-2}" <<'EOF'
import json, sys
path, k = sys.argv[1], int(sys.argv[2])
recs = []
with open(path) as f:
    for line in f:
        try:
            r = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if "eval" not in r:
            continue
        it = r.get("iter", 0)
        if it == 0:
            continue
        try:
            wr = float(r["eval"]["vs_bc"]["wr"])
        except (KeyError, TypeError, ValueError):
            continue
        recs.append((wr, it))
recs.sort(key=lambda x: (x[0], x[1]), reverse=True)
for wr, it in recs[:k]:
    print(it, wr)
EOF
)
    if [ -z "$CHOSEN" ]; then
      alert "no eval record besides iteration 0 in $OUT/rl/metrics.jsonl"
    else
      INIT_METRICS="$(dirname "$INIT")/metrics.jsonl"
      [ -f "$INIT_METRICS" ] || alert "missing init metrics $INIT_METRICS"
      CKPTS=""
      mkdir -p "$OUT/card"
      : > "$OUT/card/chosen.txt"
      while read -r it wr; do
        [ -n "$it" ] || continue
        dir=$(printf "$OUT/card/i%04d" "$it")
        mkdir -p "$dir/ckpt"
        cp "$OUT/rl/iter-$(printf "%04d" "$it").pt" "$dir/ckpt/best.pt" || { alert "missing $OUT/rl/iter-$it.pt"; continue; }
        [ -f "$INIT_METRICS" ] && cp "$INIT_METRICS" "$dir/ckpt/metrics.jsonl"
        echo "$it $wr" >> "$OUT/card/chosen.txt"
        CKPTS="$CKPTS $dir"
      done <<< "$CHOSEN"
      # shellcheck disable=SC2086
      CKPTS="$CKPTS" BASE="$INIT" PT_ARGS="${CARD_PT_ARGS:---decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards}" \
        bash scripts/vast_card.sh "$NAME/card"
      # surface card failures in the job-level ALERT the launcher relays (vast_card writes its own card/ALERT)
      [ -f "$OUT/card/ALERT" ] && grep -h . "$OUT/card/ALERT" >> "$OUT/ALERT" 2>/dev/null || true
    fi
  fi
fi
kill "$PROG_PID" 2>/dev/null
grep '^\[eval it' "$OUT/train.txt" > "$OUT/PROGRESS" 2>/dev/null || true
kill "$USAGE_PID" 2>/dev/null
echo done > "$OUT/DONE"
