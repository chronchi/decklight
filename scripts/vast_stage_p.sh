#!/usr/bin/env bash
# Retrain stage P + Pc (docs/retrain/plan-v1.md §Stage P, plan-v2 §Stage P/Pc, plan-v3 change 2026-10-13): PPO with the
# ppo1 recipe (lr 4e-5, mb 1024, p-bc 0.3 vs the frozen init, KL 0.1 to the init, new value head, GAE 0.95) from B-v1,
# a fixed 560 iterations, the last iteration is the candidate. Training decks: pool minus holdout + gauntlet (train_rl).
#   First milestone at it160 (eval_ckpt: 672 games vs B-v1, 1,488 vs ygo-agent on fixed deals, games saved):
#     stop when vs B-v1 < 0.50, mean kl_bc over its 151-160 > 0.1, or own-T1 hand-trap waste is worse than B-v1's on
#     the same deals by > 2 SE. Waste above the gate-3 bound (1.5 x ygo-agent) but not worse than B-v1 is an ALERT
#     only: B-v1 itself starts above the bound (NOTEBOOK 2026-10-13 10:30).
#   Pc card (seeds 3000+): P and base_r3 raw vs ygo-agent on the same 5,952 deals (paired), raw head-to-heads P vs
#   base_r3 on holdout (seeds 4, ~2,176 games) and new-card decks (seeds 6, ~2,184), principle audit both seats.
# Env: INIT (data/cache/bv1/best.pt), SMOKE_ITERS (6; smoke runs also set YGO_NAN_DEBUG), REF (card reference, data/cache/base_r3/best.pt; stage P3 uses P), DEAL0 (3000), ITERS (560),
#      MS_IT (160), RL_HOURS (5.5), GA_SEEDS (24), VAR_CAP (60m), SMOKE=1.
# Usage: bash scripts/vast_stage_p.sh <name>. Do NOT run on the laptop (vast GPU only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT/ms"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export HF_HOME="$PWD/data/cache/hf"
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
. scripts/cores.sh
J=$(box_cores)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }
INIT=${INIT:-data/cache/bv1/best.pt}; REF=${REF:-data/cache/base_r3/best.pt}; ITERS=${ITERS:-560}; MS_IT=${MS_IT:-160}
[ -s "$INIT" ] && [ -s "$(dirname "$INIT")/metrics.jsonl" ] || { alert "missing $INIT or its metrics.jsonl"; exit 3; }
MSE=(--eval-seeds 18 --ga-seeds 6); RLX=()
if [ "${SMOKE:-0}" = 1 ]; then ITERS=${SMOKE_ITERS:-6}; MS_IT=4; export YGO_NAN_DEBUG="$OUT/nan"; MSE=(--eval-seeds 1 --ga-seeds 1); RLX=(--rollout 4096 --envs 64 --eval-seeds 1 --value-warmup 1 --eval-every 2); fi
# train_rl's fixed eval decks (4 train + 4 holdout pool decks, random.Random(1234)): eval_ckpt needs the same names
EVN=$(uv run --no-sync python - <<'EOF'
import random
from pathlib import Path
hold = set(Path("decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
pool = sorted(Path("decks/pool").glob("*.ydk"))
r = random.Random(1234)
names = r.sample([p.stem for p in pool if p.stem not in hold], 4) + r.sample(sorted(hold & {p.stem for p in pool}), 4)
print(",".join(names))
EOF
)
[ -n "$EVN" ] || { alert "no eval deck names"; exit 3; }
evalck() {  # <ckpt> <iter> <tag>: milestone-size eval with saved games
  uv run --no-sync python scripts/eval_ckpt.py --ckpt "$1" --init "$INIT" --iter "$2" --eval-names "$EVN" "${MSE[@]}" \
    --threads 6 --out "$OUT/ms/$3.jsonl" --games-dir "$OUT/ms/$3" > "$OUT/ms/$3.txt" 2>&1
}
waste() {  # <tag>: own-T1 hand-trap games of the pilot (s) and ygo-agent (ya) in the saved vs_ygoagent games
  for s in s ya; do
    uv run --no-sync python scripts/audit/principles.py --name "$1-$s" --games "$OUT/ms/$1/vs_ygoagent" --seat $s \
      --out "$OUT/ms/principles-$1-$s.json" > "$OUT/ms/principles-$1-$s.txt" 2>&1 || return 1
  done
}

# B-v1's waste on the milestone deals (the reference for the it160 check), run while PPO starts
( evalck "$INIT" 0 init && waste init || alert "milestone reference (B-v1) eval/audit failed: $(tail -1 "$OUT/ms/init.txt")" ) &
REFP=$!

# 1. PPO
touch "$OUT/train.txt"
uv run --no-sync python scripts/train_rl.py --init "$INIT" --out "$OUT/rl" --device cuda --lr 4e-5 --iters "$ITERS" \
  --hours "${RL_HOURS:-5.5}" --eval-every 40 "${RLX[@]}" > "$OUT/train.txt" 2>&1 &
TP=$!
milestone() {
  local ck; ck=$(printf "%s/rl/iter-%04d.pt" "$OUT" "$MS_IT")
  evalck "$ck" "$MS_IT" "i$MS_IT" && waste "i$MS_IT" || { alert "milestone eval/audit failed: $(tail -1 "$OUT/ms/i$MS_IT.txt")"; return 1; }
  wait $REFP
  uv run --no-sync python - "$OUT" "$MS_IT" <<'EOF'
import json, math, sys
from pathlib import Path
o, it = Path(sys.argv[1]), int(sys.argv[2])
ev = json.loads((o / f"ms/i{it}.jsonl").read_text().splitlines()[-1])["eval"]
recs = [json.loads(x) for x in (o / "rl/metrics.jsonl").read_text().splitlines() if '"kl_bc"' in x]
kl = [r["kl_bc"] for r in recs if it - 10 < r["iter"] <= it]
p = {s: json.loads((o / f"ms/principles-i{it}-{s}.json").read_text()) for s in ("s", "ya")}
ref = {s: json.loads((o / f"ms/principles-init-{s}.json").read_text()) for s in ("s", "ya")}
pw, yw, bw = p["s"]["lines_with_own_t1_ht"], p["ya"]["lines_with_own_t1_ht"], ref["s"]["lines_with_own_t1_ht"]
bound = 1.5 * yw
kl_m = sum(kl) / len(kl) if kl else float("nan")
print(f"milestone it{it}: vs init {ev['vs_bc']['wr']} n={ev['vs_bc']['n']} {ev['vs_bc']['ci']}; vs ygo-agent "
      f"{ev['vs_ygoagent']['wr']} n={ev['vs_ygoagent']['n']}; kl_bc mean {kl_m:.4f}; own-T1 HT games cand {pw} vs init {bw} "
      f"(ygo-agent {yw}, bound {bound:.0f}); interrupts P {p['s']['interrupt_opp']['rate']} ygo-agent {p['ya']['interrupt_opp']['rate']}")
bad = []
if ev["vs_bc"]["wr"] < 0.50:
    bad.append(f"vs init {ev['vs_bc']['wr']} < 0.50")
if not kl or kl_m > 0.1:
    bad.append(f"kl_bc mean {kl_m:.4f} > 0.1 (or missing)")
if pw > bw + 2 * math.sqrt(max(bw, 1)):
    bad.append(f"own-T1 waste worse than init ({pw} vs {bw})")
if pw > bound:
    print(f"ALERT milestone: own-T1 HT games {pw} above the gate-3 bound {bound:.0f} (B-v1 {bw}; not a stop)")
for b in bad:
    print(f"ALERT milestone: {b}")
sys.exit(1 if bad else 0)
EOF
}
( done_ms=0; seen=0; while kill -0 $TP 2>/dev/null; do
    sleep 30
    n=$(grep -c '^\[eval it' "$OUT/train.txt" 2>/dev/null); n=${n:-0}
    if [ "$n" -gt "$seen" ]; then
      grep '^\[eval it' "$OUT/train.txt" | tail -n +$((seen + 1)) | sed "s/^/$(date +%T) /" >> "$OUT/PROGRESS"; seen=$n
    fi
    [ $done_ms = 0 ] && [ -f "$(printf "%s/rl/iter-%04d.pt" "$OUT" "$MS_IT")" ] || continue
    done_ms=1
    milestone > "$OUT/milestone.txt" 2>&1; rc=$?
    echo "$(date +%T) $(grep '^milestone' "$OUT/milestone.txt")" >> "$OUT/PROGRESS"
    grep -h '^ALERT' "$OUT/milestone.txt" >> "$OUT/ALERT"
    if [ $rc -ne 0 ] && [ "${SMOKE:-0}" != 1 ]; then
      alert "first milestone failed: PPO stopped"; touch "$OUT/MILESTONE_FAILED"; pkill -TERM -P $TP; kill $TP
    fi
  done ) &
WP=$!
wait $TP; rc=$?
wait $WP
grep -h "^ALERT\|Traceback" "$OUT/train.txt" | head -5 >> "$OUT/ALERT"
# keep the critic (fp16 weights incl. value_head) before dropping last.pt's optimizer state: stage P 2026-10-13 deleted
# P's only critic this way (NOTEBOOK 23:30)
[ -s "$OUT/rl/last.pt" ] && uv run --no-sync python -c "
import torch, sys
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)['model']
torch.save({k: v.half() if v.is_floating_point() else v for k, v in ck.items()}, sys.argv[2])" \
  "$OUT/rl/last.pt" "$OUT/rl/last_with_value.pt" && rm -f "$OUT/rl/last.pt" || alert "critic export failed; last.pt kept"
[ -f "$OUT/MILESTONE_FAILED" ] && exit 4
[ -f "$OUT/milestone.txt" ] || alert "milestone never ran (no iter-$MS_IT checkpoint)"
LAST=$(ls "$OUT"/rl/iter-*.pt 2>/dev/null | sort | tail -1)
if [ $rc -ne 0 ] || [ -z "$LAST" ]; then alert "train_rl failed (exit $rc): $(tail -1 "$OUT/train.txt")"; exit 3; fi
[ "$LAST" = "$(printf "%s/rl/iter-%04d.pt" "$OUT" "$ITERS")" ] || alert "last checkpoint is $LAST, not iteration $ITERS"
mkdir -p "$OUT/P"; cp "$LAST" "$OUT/P/best.pt"; cp "$(dirname "$INIT")/metrics.jsonl" "$OUT/P/metrics.jsonl"
cmp -s "$OUT/P/best.pt" "$INIT" && { alert "candidate equals the init weights"; exit 3; }
tick "ppo $(basename "$LAST")"

# 2. Pc card on fresh deals (seeds 3000+)
GS=${GA_SEEDS:-24}; HS=4; NS=6; [ "${SMOKE:-0}" = 1 ] && { GS=1; HS=1; NS=1; }
V="p|--opp ygoagent --raw 1 --ckpt $OUT/P/best.pt"
V="$V;base|--opp ygoagent --raw 1 --ckpt $REF"
V="$V;p-hold|--opp ckpt --opp-ckpt $REF --raw 1 --deck-set holdout --seeds $HS --ckpt $OUT/P/best.pt"
V="$V;p-new|--opp ckpt --opp-ckpt $REF --raw 1 --deck-set newcards --seeds $NS --ckpt $OUT/P/best.pt"
CKPT=$REF SM_ARGS="--ga-seeds $GS --deal-seed0 ${DEAL0:-3000}" VARIANTS="$V" PROBE_NPROC=1 \
  VAR_CAP="${VAR_CAP:-60m}" bash scripts/vast_search_match.sh "$NAME"
tick card
uv run --no-sync python scripts/audit/paired_sm.py "$OUT/base" "$OUT/p" > "$OUT/paired-base.txt" 2>&1 \
  || alert "paired_sm: $(tail -1 "$OUT/paired-base.txt")"
echo "$(date +%T) paired P - base_r3: $(sed -n 2,4p "$OUT/paired-base.txt" | tr '\n' ' ')" >> "$OUT/PROGRESS"
for t in p:s p:ya base:s; do
  uv run --no-sync python scripts/audit/principles.py --name "${t/:/-}" --games "$OUT/${t%%:*}" --seat "${t##*:}" \
    --out "$OUT/principles-${t/:/-}.json" > "$OUT/principles-${t/:/-}.txt" 2>&1 || alert "principles $t failed"
done
uv run --no-sync python - "$OUT" >> "$OUT/PROGRESS" <<'EOF'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
pr = {k: json.loads((o / f"principles-{k}.json").read_text()) for k in ("p-s", "p-ya", "base-s")}
t1 = {k: v["lines_with_own_t1_ht"] for k, v in pr.items()}
io = {k: v["interrupt_opp"]["rate"] for k, v in pr.items()}
print(f"Pc behaviour: own-T1 HT games P {t1['p-s']} base_r3 {t1['base-s']} ygo-agent {t1['p-ya']} (bound {1.5 * t1['p-ya']:.0f}); "
      f"interrupts P {io['p-s']} base_r3 {io['base-s']} ygo-agent {io['p-ya']} (ratio {io['p-s'] / io['p-ya']:.2f})")
for line in (o / "paired-base.txt").read_text().splitlines()[1:]:
    if "SE" in line:  # 90% lower bound for the non-inferiority ladder (plan-v2 Pc: >= -2.5 pp, both seats)
        m = float(line.split("B-A ")[1].split(" pp")[0]); se = float(line.split("SE ")[1].split(")")[0])
        print(f"Pc ladder {line.split(':')[0]}: P - base_r3 {m:+.1f} pp, lower 90% {m - 1.645 * se:+.1f}")
EOF
true
