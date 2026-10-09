#!/usr/bin/env bash
# Bank learnability diagnostic (scripts/bank_ft.py, NOTEBOOK 2026-10-09): fine-tune INIT on the hint bank's train starts
# (GPU), play every banked start with base / ft / ftfinal vs OPP and OPP2 (CPU, sharded by start), paired report,
# then the standard report card of ft vs INIT (vast_card.sh, CARD_PT_ARGS; CARD=0 skips).
# Env: INIT, BANK, OPP, OPP2, EPOCHS (12), LR (1e-4), GAMES (8), HELD (0.25), CARD_PT_ARGS, CARD (1).
# Usage: bash scripts/vast_bank_ft.sh <name>. Results in runs/<name>/: bankft/{split.json, ft/metrics.jsonl,
# report.json, play-*.jsonl}, train.txt, play.txt, report.txt, card/, ALERT. Do NOT run on the laptop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
: "${INIT:?}" "${BANK:?}" "${OPP:?}"
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
if [ "$Q" = "max" ]; then J=$(nproc); else J=$(( Q / P )); fi
[ "$J" -lt 2 ] && J=2
export HF_HOME="$PWD/data/cache/hf"
T0=$(date +%s)
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
echo "deps install $(( $(date +%s) - T0 ))s" >> "$OUT/torch.txt"
{ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo "jobs $J"; } > "$OUT/machine.txt" 2>&1
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg)"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
T0=$SECONDS
tick() { echo "$1 $((SECONDS - T0)) s" >> "$OUT/timing.txt"; }
D="$OUT/bankft"
uv run --no-sync python scripts/bank_ft.py train --bank "$BANK" --init "$INIT" --out "$D" --device cuda \
  --epochs "${EPOCHS:-12}" --lr "${LR:-1e-4}" --held-frac "${HELD:-0.25}" > "$OUT/train.txt" 2>&1 \
  || { alert "bank_ft train failed: $(tail -1 "$OUT/train.txt")"; kill "$USAGE_PID"; echo done > "$OUT/DONE"; exit 1; }
tick train
grep -h "^ALERT" "$OUT/train.txt" >> "$OUT/ALERT" 2>/dev/null
# milestone check: the fine-tune must fit its own train decisions (train CE falls below the init's)
uv run --no-sync python - "$D/ft/metrics.jsonl" <<'PY' || alert "train CE did not fall: the fine-tune did not fit"
import json, sys
r = [json.loads(x) for x in open(sys.argv[1]) if x.strip()][1:]
i, f = r[0], r[-1]
print(f"init train {i['train']} held {i['held']} -> final train {f['train']} held {f['held']}")
sys.exit(0 if f["train"]["ce"] < i["train"]["ce"] else 1)
PY
NS=$(uv run --no-sync python -c "import json,sys; s=json.load(open('$D/split.json')); print(len(s['train'])+len(s['held']))")
NP=$(( J < NS ? J : NS ))
PIDS=()
for p in $(seq 0 $((NP - 1))); do
  uv run --no-sync python scripts/bank_ft.py play --out "$D" --opp "$OPP" ${OPP2:+--opp2 "$OPP2"} \
    --games "${GAMES:-8}" --nproc "$NP" --proc "$p" > "$OUT/play-$p.txt" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || alert "play pid $pid exit $?"; done
cat "$OUT"/play-*.txt > "$OUT/play.txt"; rm -f "$OUT"/play-[0-9]*.txt
grep -h "^ALERT\|Traceback" "$OUT/play.txt" >> "$OUT/ALERT" 2>/dev/null
tick play
uv run --no-sync python scripts/bank_ft.py report --out "$D" > "$OUT/report.txt" 2>&1 || alert "report failed: $(tail -1 "$OUT/report.txt")"
tick report
if [ "${CARD:-1}" != "0" ]; then
  C="$OUT/cardin/ft"; mkdir -p "$C/ckpt"
  cp "$D/ft/best.pt" "$C/ckpt/best.pt" && cp "$D/ft/metrics.jsonl" "$C/ckpt/metrics.jsonl"
  CKPTS="$C" BASE="$INIT" PT_ARGS="${CARD_PT_ARGS:---decks 8 --hands 3 --seeds 8 --narrate 0 --samples 4 --ygoagent --ga-seeds 3 --newcards decks/newcards}" \
    bash scripts/vast_card.sh "$NAME/card"
  [ -f "$OUT/card/ALERT" ] && grep -h . "$OUT/card/ALERT" >> "$OUT/ALERT" 2>/dev/null || true
  rm -f "$C/ckpt/best.pt"  # the same weights stay in bankft/ft/best.pt
  tick card
fi
kill "$USAGE_PID" 2>/dev/null
echo done > "$OUT/DONE"
