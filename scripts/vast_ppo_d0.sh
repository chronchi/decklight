#!/usr/bin/env bash
# Step A (D0, plan-v4 §6; user 2026-10-15 "Yes" to "start with step A"): continue PPO from P. Does P have room left?
#   1. PPO: stage P's recipe (lr 4e-5, mb 1024, league 0 = self-play + 0.3 vs the frozen init, KL 0.1 to the init),
#      init AND anchor = P, value head warm from value_p (--value-init), ITERS iterations. No ygo-agent episodes in the
#      rollouts (user 2026-10-15: "careful to not train our models just to beat ygo-agents").
#      First milestone (it MS_IT): train_rl's eval vs the frozen P (4 train + 4 holdout decks) >= 0.45 and mean kl_bc
#      over the last 10 iterations <= 0.1, else stop. last.pt is kept so a pass can be extended with --resume (A+).
#   2. card: P raw and P' raw vs ygo-agent on the same 5,952 FRESH deals (deal seeds 8000+), paired; h2h P' vs P on
#      holdout (seeds 4) and new-card decks (seeds 6), both seats; principles for P'.
#   Gate (pre-registered, NOTEBOOK 2026-10-15): P' - P >= +2 pp paired vs ygo-agent AND unseen h2h vs P (holdout +
#   new-card pooled) >= 0.50. Unseen h2h >= 0.53 with < +2 pp vs ygo-agent is reported as "stronger off-benchmark".
# Env: ITERS (160), MS_IT (80), RL_HOURS (2.5), GA_SEEDS (24), VAR_CAP (60m), SMOKE=1, RL_EXTRA (extra train_rl
#      flags), DEAL0 (8000), CTRL (X1, 2026-10-27: a stored control checkpoint; the card adds it raw vs ygo-agent on
#      the same deals, paired against P' in paired-ctrl.txt, and the unseen h2h opponent becomes CTRL instead of P).
# Usage: bash scripts/vast_ppo_d0.sh <name>. vast GPU only.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export HF_HOME="$PWD/data/cache/hf"
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }
P=data/cache/P/best.pt; VP=data/cache/value_p/best.pt
for f in "$P" data/cache/P/metrics.jsonl "$VP"; do [ -s "$f" ] || { alert "missing $f"; exit 3; }; done
ITERS=${ITERS:-160}; MS_IT=${MS_IT:-80}; GS=${GA_SEEDS:-24}; HS=4; NS=6; RLX=()
if [ "${SMOKE:-0}" = 1 ]; then ITERS=4; MS_IT=2; GS=1; HS=1; NS=1
  RLX=(--rollout 4096 --envs 64 --eval-seeds 1 --eval-every 2); fi

# 1. PPO from P
uv run --no-sync python scripts/train_rl.py --init "$P" --value-init "$VP" --out "$OUT/rl" --device cuda --lr 4e-5 \
  --iters "$ITERS" --hours "${RL_HOURS:-2.5}" --eval-every 40 "${RLX[@]}" ${RL_EXTRA:-} > "$OUT/train.txt" 2>&1 &
TP=$!
( seen=0; seen_a=0; done_ms=0; while kill -0 $TP 2>/dev/null; do
    sleep 30
    n=$(grep -c '^\[eval it' "$OUT/train.txt" 2>/dev/null); n=${n:-0}
    if [ "$n" -gt "$seen" ]; then
      grep '^\[eval it' "$OUT/train.txt" | tail -n +$((seen + 1)) | sed "s/^/$(date +%T) /" >> "$OUT/PROGRESS"; seen=$n
    fi
    na=$(grep -c '^ALERT' "$OUT/train.txt" 2>/dev/null); na=${na:-0}  # train_rl's own ALERTs (adv_filter), live
    if [ "$na" -gt "$seen_a" ]; then grep '^ALERT' "$OUT/train.txt" | tail -n +$((seen_a + 1)) >> "$OUT/ALERT"; seen_a=$na; fi
    [ $done_ms = 0 ] && grep -q "^\[eval it $MS_IT\]" "$OUT/train.txt" || continue
    done_ms=1
    uv run --no-sync python - "$OUT" "$MS_IT" > "$OUT/milestone.txt" 2>&1 <<'EOF'
import json, sys
from pathlib import Path
o, it = Path(sys.argv[1]), int(sys.argv[2])
recs = [json.loads(x) for x in (o / "rl/metrics.jsonl").read_text().splitlines() if x.strip()]
ev = [r["eval"] for r in recs if r.get("iter") == it and "eval" in r]
kl = [r["kl_bc"] for r in recs if "kl_bc" in r and it - 10 < r["iter"] <= it]
vs = ev[-1]["vs_bc"]["wr"] if ev else float("nan")
kl_m = sum(kl) / len(kl) if kl else float("nan")
hv = ev[-1]["vs_bc_holdout"]["wr"] if ev else float("nan")
keep = [r["adv_keep"] for r in recs if "adv_keep" in r and it - 10 < r["iter"] <= it]
thr = [r["adv_thr"] for r in recs if "adv_thr" in r and it - 10 < r["iter"] <= it]
af = f"; adv_keep mean {sum(keep) / len(keep):.3f}, adv_thr mean {sum(thr) / len(thr):.4f}" if keep else ""
print(f"milestone it{it}: vs frozen P {vs} (holdout decks {hv}); kl_bc mean {kl_m:.4f}{af}")
bad = [b for b, c in ((f"vs frozen P {vs} < 0.45", not vs >= 0.45), (f"kl_bc mean {kl_m:.4f} > 0.1", not kl_m <= 0.1)) if c]
for b in bad:
    print(f"ALERT milestone: {b}")
sys.exit(1 if bad else 0)
EOF
    rc=$?
    case "${RL_EXTRA:-}" in *--adv-filter-q*) grep -q "adv_keep mean" "$OUT/milestone.txt" \
      || { echo "ALERT milestone: --adv-filter-q set but no adv_keep in metrics" >> "$OUT/milestone.txt"; rc=1; } ;; esac
    echo "$(date +%T) $(grep '^milestone' "$OUT/milestone.txt")" >> "$OUT/PROGRESS"
    grep -h '^ALERT' "$OUT/milestone.txt" >> "$OUT/ALERT"
    if [ $rc -ne 0 ] && [ "${SMOKE:-0}" != 1 ]; then
      alert "first milestone failed: PPO stopped ($(tail -3 "$OUT/milestone.txt" | tr '\n' ' '))"
      touch "$OUT/MILESTONE_FAILED"; pkill -TERM -P $TP; kill $TP
    fi
  done ) &
WP=$!
wait $TP; rc=$?
wait $WP
grep -h "^ALERT\|Traceback" "$OUT/train.txt" | head -5 >> "$OUT/ALERT"
[ -f "$OUT/MILESTONE_FAILED" ] && exit 4
LAST=$(ls "$OUT"/rl/iter-*.pt 2>/dev/null | sort | tail -1)
if [ $rc -ne 0 ] || [ -z "$LAST" ]; then alert "train_rl failed (exit $rc): $(tail -1 "$OUT/train.txt")"; exit 3; fi
[ "$LAST" = "$(printf "%s/rl/iter-%04d.pt" "$OUT" "$ITERS")" ] || alert "last checkpoint is $LAST, not iteration $ITERS"
mkdir -p "$OUT/Pp"; cp "$LAST" "$OUT/Pp/best.pt"; cp data/cache/P/metrics.jsonl "$OUT/Pp/metrics.jsonl"
cmp -s "$OUT/Pp/best.pt" "$P" && { alert "P' equals P's weights"; exit 3; }
tick "ppo $(basename "$LAST")"

# 2. card on fresh deals (deal seeds DEAL0+; 8000 for D0)
PP="$OUT/Pp/best.pt"; H2H="${CTRL:-$P}"
if [ -n "${CTRL:-}" ]; then
  [ -s "$CTRL" ] || { alert "missing CTRL $CTRL"; exit 3; }
  cmp -s "$CTRL" "$PP" && { alert "P' equals CTRL's weights"; exit 3; }
fi
V="praw|--opp ygoagent --raw 1"
V="$V;pp|--opp ygoagent --raw 1 --ckpt $PP"
[ -n "${CTRL:-}" ] && V="$V;ctrl|--opp ygoagent --raw 1 --ckpt $CTRL"
V="$V;pp-hold|--opp ckpt --opp-ckpt $H2H --raw 1 --deck-set holdout --seeds $HS --ckpt $PP"
V="$V;pp-new|--opp ckpt --opp-ckpt $H2H --raw 1 --deck-set newcards --seeds $NS --ckpt $PP"
CKPT="$P" SM_ARGS="--ga-seeds $GS --deal-seed0 ${DEAL0:-8000}" VARIANTS="$V" PROBE_NPROC=1 VAR_CAP="${VAR_CAP:-60m}" \
  bash scripts/vast_search_match.sh "$NAME"
tick card
uv run --no-sync python scripts/audit/paired_sm.py "$OUT/praw" "$OUT/pp" > "$OUT/paired.txt" 2>&1 \
  || alert "paired_sm: $(tail -1 "$OUT/paired.txt")"
echo "$(date +%T) paired P' - P: $(sed -n 1,4p "$OUT/paired.txt" | tr '\n' ' ')" >> "$OUT/PROGRESS"
if [ -n "${CTRL:-}" ]; then
  uv run --no-sync python scripts/audit/paired_sm.py "$OUT/ctrl" "$OUT/pp" > "$OUT/paired-ctrl.txt" 2>&1 \
    || alert "paired_sm ctrl: $(tail -1 "$OUT/paired-ctrl.txt")"
  echo "$(date +%T) paired P' - CTRL: $(sed -n 1,4p "$OUT/paired-ctrl.txt" | tr '\n' ' ')" >> "$OUT/PROGRESS"
fi
for s in s ya; do
  uv run --no-sync python scripts/audit/principles.py --name "pp-$s" --games "$OUT/pp" --seat $s \
    --out "$OUT/principles-pp-$s.json" > "$OUT/principles-pp-$s.txt" 2>&1 || alert "principles $s failed"
done
uv run --no-sync python - "$OUT" <<'EOF' >> "$OUT/PROGRESS"
import json, sys
o = sys.argv[1]
s, ya = (json.load(open(f"{o}/principles-pp-{x}.json")) for x in ("s", "ya"))
print(f"P' behaviour: own-T1 HT games {s['lines_with_own_t1_ht']} (gate <= 1.5 x ygo-agent {ya['lines_with_own_t1_ht']}); "
      f"interrupts {s['interrupt_opp']['rate']} vs ygo-agent {ya['interrupt_opp']['rate']}")
EOF
true
