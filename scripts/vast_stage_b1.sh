#!/usr/bin/env bash
# Retrain stage B1 (docs/retrain/plan-v2.md §Stage B1, plan-v3): the data decision on the OLD architecture. Two B-v1s
# from scratch with base_r3's architecture (train_bc defaults d256 / 4 layers / 4 heads):
#   clean = the stage D clean mix (plan-v1 §3.4 with plan-v2 §3 changes) + stage M turns;
#   hw    = clean + the stage L decisive window labels (data/demos/sw-hw, kinds sw-hw / sw-hw_own, weight 1).
# Then the card: every policy raw vs ygo-agent on the same 5,952 fresh deals (31 lists x 4 decks x 24 seeds x 2 seats,
# deal seeds 2000+): B-clean, B-hw, ft-teach-h, ft-swv, base_r3; plus raw head-to-head of each B-v1 vs ft-swv on the
# holdout and new-card decks.
# Pre-registered decision: higher card score goes forward (< 1.5 pp apart: better behaviour bands); go/no-go: the chosen
# B-v1 >= ft-swv - 3 pp, else stop and ask the user before B.
# Env: EPOCHS (6), ANCHOR_DROPS (jsonl of stage-M turn line ids the play-out anchor dropped; required unless
# NO_TURNS=1), VARIANTS ("clean hw"), HW_FRAC (per-epoch repeat of the hw labels, default 1 = plan-v2 weight 1; 1,483
# labels are ~0.07% of the mix, NOTEBOOK 2026-10-12 06:20), GA_SEEDS (24), VAR_CAP (60m), SMOKE=1 (tiny everything).
# Pass-weight test (NOTEBOOK 2026-10-13 19:00): EXTRA_FRAC (more --kind-frac entries, e.g. "teacher_pass=2 dagger_pass=2"),
# CARD=0 (skip the 5,952-deal card), MS_EVAL=1 (instead: P's milestone eval of clean/best.pt vs B-v1 at
# data/cache/bv1 on the same 1,488 ygo-agent deals + principles on both seats, comparable with runs/stage-p-*/ms/init).
# Usage: bash scripts/vast_stage_b1.sh <name>. Do NOT run on the laptop (vast GPU only; ~45 GB RAM per training).
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
. scripts/cores.sh
J=$(box_cores)
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }
EPOCHS=${EPOCHS:-6}; VARIANTS=" ${VARIANTS:-clean hw} "

# 1. shards: strict labels (inferred -> weight 0) + the stage D exclusions (docs/retrain/recipe.md) + anchor drops
A=data/cache/audit; OLD='data/demos/*,!*/strict-d/*,!*/guides-strict-d/*'
EXCL=("$OLD@$A/strict_junk.json" "data/demos/planned*,!*/strict-d/*@$A/drops-muse.final.jsonl"
      "data/demos/claude/*,!*/strict-d/*@$A/drops-claude.final.jsonl"
      "data/demos/verified/*,data/demos/expanded/*@$A/drops-guides.final.jsonl"
      "*/strict-d/*,*/guides-strict-d/*@$A/drops-strictd.final.jsonl")
if [ "${NO_TURNS:-0}" != 1 ]; then
  [ -s "${ANCHOR_DROPS:-}" ] || { alert "ANCHOR_DROPS missing (${ANCHOR_DROPS:-unset}); set NO_TURNS=1 to train without stage M"; exit 3; }
  EXCL+=("data/demos/turns/*@$ANCHOR_DROPS")
fi
for f in $A/strict_junk.json $A/drops-muse.final.jsonl $A/drops-claude.final.jsonl $A/drops-guides.final.jsonl \
         $A/drops-strictd.final.jsonl; do [ -s "$f" ] || { alert "missing exclusion file $f"; exit 3; }; done
SONLY=(--only guides claude muse gold teacher dagger sw turns)
[ "${SMOKE:-0}" = 1 ] && SONLY=(--only gold dagger sw turns)
uv run --no-sync python scripts/build_bc_shards.py "${SONLY[@]}" --strict-labels --out data/cache/bc_shards --jobs "$J" \
  --exclude "${EXCL[@]}" > "$OUT/shards.txt" 2>&1 || { alert "shards failed: $(tail -1 "$OUT/shards.txt")"; exit 3; }
grep -h "^ALERT" "$OUT/shards.txt" >> "$OUT/ALERT"
tick "shards $(grep -m1 '^samples' "$OUT/shards.txt")"
# per-kind sample counts (the mix is only as good as what reached the shards; 0 of a kind we train on is a bug)
uv run --no-sync python - "$OUT" <<'EOF' || { alert "shard kind check failed: $(tail -1 "$OUT/kinds.txt")"; exit 3; }
import sys, collections, numpy as np
from pathlib import Path
c, w = collections.Counter(), collections.defaultdict(float)
for f in sorted(Path("data/cache/bc_shards").glob("*.npz")):  # only the kind/weight members (obs would be ~45 GB)
    z = np.load(f)
    if "kind" not in z.files:
        continue
    for k, wt in zip(z["kind"].tolist(), z["weight"].tolist()):
        c[k] += 1
        w[k] += float(wt)
lines = [f"{k}\t{n}\tweighted {w[k]:.0f}" for k, n in sorted(c.items())]
Path(sys.argv[1], "kinds.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
import os
need = ["dagger", "sw-hw", "sw-hw_own"] if os.environ.get("SMOKE") == "1" else ["teacher", "dagger", "line", "sw-hw", "sw-hw_own"]
missing = [k for k in need if w.get(k, 0) == 0]
if missing:
    print(f"ALERT 0 weighted samples of {missing}")
    sys.exit(1)
EOF

# 2. two B-v1s from scratch, sequential (RAM). Mix = plan-v1 §3.4 with plan-v2 §3: gold_pass 0, inferred PASS 0 (strict
# labels), hw 1 if picked; stage M turns at line weight. Kinds not listed are not trained (games/, search, selfimit,
# window, sw-v and the other old sw-* labellings are dropped by plan-v1 §3.2).
HOLD="kewl-tune $( [ -f decks/pool/HOLDOUT ] && tr '\n' ' ' < decks/pool/HOLDOUT )"
CLEAN=(teacher teacher_pass dagger dagger_pass line line_improved line_expanded gold turn_mid turn_second turn_end)
# pass kinds already carry weight 0.5 in the shards (smoke 2026-10-12: dagger_pass 46,608 samples, weighted 23,304),
# so plan-v1 §3.4 teacher_pass / dagger_pass 0.5 needs no extra fraction
# shellcheck disable=SC2206
FRAC=(line_expanded=0.3 gold=3 ${EXTRA_FRAC:-})
EXTRA=(); [ "${SMOKE:-0}" = 1 ] && EXTRA=(--limit 20000)
V="ftswv|--opp ygoagent --raw 1 --ckpt data/cache/base_r2/best.pt"
V="$V;teachh|--opp ygoagent --raw 1 --ckpt data/cache/base_teach_h/best.pt"
V="$V;base|--opp ygoagent --raw 1 --ckpt data/cache/base_r3/best.pt"
for v in clean hw; do
  [[ "$VARIANTS" == *" $v "* ]] || continue
  K=("${CLEAN[@]}"); F=("${FRAC[@]}")
  [ "$v" = hw ] && { K+=(sw-hw sw-hw_own); F+=(sw-hw="${HW_FRAC:-1}" sw-hw_own="${HW_FRAC:-1}"); }
  # shellcheck disable=SC2086
  uv run --no-sync python scripts/train_bc.py --shards data/cache/bc_shards --emb data/cards/emb/qwen3-0.6b-d512.npz \
    --out "$OUT/$v" --epochs "$EPOCHS" --batch 512 --lr 5e-4 --holdout $HOLD --select-on last --kinds "${K[@]}" \
    --kind-frac "${F[@]}" "${EXTRA[@]}" > "$OUT/train-$v.txt" 2>&1 &
  TP=$!
  # relay each epoch's metrics line to PROGRESS (first-milestone read: does val move the intended way after epoch 0)
  ( seen=0; while kill -0 $TP 2>/dev/null; do
      sleep 60; n=$(grep -c '"epoch"' "$OUT/$v/metrics.jsonl" 2>/dev/null); n=${n:-0}
      if [ "$n" -gt "$seen" ]; then echo "$(date +%T) train-$v $(tail -1 "$OUT/$v/metrics.jsonl" | cut -c1-300)" >> "$OUT/PROGRESS"; seen=$n; fi
    done ) &
  wait $TP; rc=$?
  rm -f "$OUT/$v/last.pt"
  grep -h "^ALERT" "$OUT/train-$v.txt" >> "$OUT/ALERT"
  if [ $rc -ne 0 ] || [ ! -s "$OUT/$v/best.pt" ]; then alert "train-$v failed (exit $rc): $(tail -1 "$OUT/train-$v.txt")"; continue; fi
  tick "train-$v"
  V="$V;b$v|--opp ygoagent --raw 1 --ckpt $OUT/$v/best.pt"
  V="$V;b$v-hold|--opp ckpt --opp-ckpt data/cache/base_r2/best.pt --raw 1 --deck-set holdout --seeds ${HOLD_SEEDS:-2} --ckpt $OUT/$v/best.pt"
  V="$V;b$v-new|--opp ckpt --opp-ckpt data/cache/base_r2/best.pt --raw 1 --deck-set newcards --seeds ${NEW_SEEDS:-3} --ckpt $OUT/$v/best.pt"
done
ls "$OUT"/*/best.pt > /dev/null 2>&1 || { alert "no B-v1 trained"; exit 3; }

if [ "${MS_EVAL:-0}" = 1 ]; then  # P's milestone eval (vast_stage_p.sh evalck/waste), same decks, seeds and deals
  EVN=$(uv run --no-sync python - <<'PY'
import random
from pathlib import Path
hold = set(Path("decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
pool = sorted(Path("decks/pool").glob("*.ydk"))
r = random.Random(1234)
names = r.sample([p.stem for p in pool if p.stem not in hold], 4) + r.sample(sorted(hold & {p.stem for p in pool}), 4)
print(",".join(names))
PY
)
  MSE=(--eval-seeds 18 --ga-seeds 6); [ "${SMOKE:-0}" = 1 ] && MSE=(--eval-seeds 1 --ga-seeds 1)
  mkdir -p "$OUT/ms"
  uv run --no-sync python scripts/eval_ckpt.py --ckpt "$OUT/clean/best.pt" --init data/cache/bv1/best.pt --iter 0 \
    --eval-names "$EVN" "${MSE[@]}" --threads 6 --out "$OUT/ms/pw.jsonl" --games-dir "$OUT/ms/pw" > "$OUT/ms/pw.txt" 2>&1 \
    || { alert "eval failed: $(tail -1 "$OUT/ms/pw.txt")"; exit 4; }
  echo "$(date +%T) eval $(grep '\[eval' "$OUT/ms/pw.txt" | tail -1)" >> "$OUT/PROGRESS"
  for s in s ya; do
    uv run --no-sync python scripts/audit/principles.py --name "pw-$s" --games "$OUT/ms/pw/vs_ygoagent" --seat $s \
      --out "$OUT/ms/principles-pw-$s.json" > "$OUT/ms/principles-pw-$s.txt" 2>&1 || { alert "principles $s failed"; exit 4; }
  done
  uv run --no-sync python - "$OUT" <<'PY' >> "$OUT/PROGRESS"
import json, sys
o = sys.argv[1]
s, ya = (json.load(open(f"{o}/ms/principles-pw-{x}.json")) for x in ("s", "ya"))
e = s["empty_draw_own"]
maxx = s["card_counts"]["ht_own_t1"].get("23434538", 0)
print(f"pass-weight: empty_draw_own {e['fires']}/{e['opps']} = {e['rate']} (B-v1 0.568, gate <= 0.40); own-T1 HT games "
      f"{s['lines_with_own_t1_ht']} (B-v1 115, gate <= 1.5 x ygo-agent {ya['lines_with_own_t1_ht']}); own-T1 Maxx {maxx} (B-v1 59)")
PY
fi
[ "${CARD:-1}" = 0 ] && exit 0

# 3. the card: every arm raw on the same fresh deals (paired later with scripts/audit/paired_gate.py)
GS=${GA_SEEDS:-24}; [ "${SMOKE:-0}" = 1 ] && GS=1
CKPT=data/cache/base_r3/best.pt SM_ARGS="--ga-seeds $GS --deal-seed0 2000" VARIANTS="$V" PROBE_NPROC=1 \
  VAR_CAP="${VAR_CAP:-60m}" bash scripts/vast_search_match.sh "$NAME"
tick card
for t in ftswv teachh base bclean bhw; do
  [ -f "$OUT/$t/summary.json" ] && echo "card $t $(tr -d '\n ' < "$OUT/$t/summary.json" | cut -c1-200)" >> "$OUT/PROGRESS"
done
true  # the last arm may be absent (B1 ran clean only, user 2026-10-12): not a job failure
