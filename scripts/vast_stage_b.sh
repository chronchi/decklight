#!/usr/bin/env bash
# Retrain stage B (docs/retrain/plan-v2.md §Stage B, plan-v3): B-v3 = PolicyNetV3 (effect strings + Lua records, long
# salient history) from scratch on B1's chosen mix (the clean arm, NOTEBOOK 2026-10-13 B1) + every anchored stage M turn.
#   1. shards exactly as B1 (strict labels + stage D exclusions + ANCHOR_DROPS);
#   2. train 6 epochs, --select-on last (the candidate is the last epoch). First milestone after epoch 1 (plan-v2):
#      loss finite, val teacher agreement rising, and the context-behaviour band (scripts/audit/ctx_band.py on ep1.pt,
#      reference: B-v1 on the same states). A failed milestone kills the training and skips the card.
#   3. the card: B-v3 raw vs ygo-agent on B1's 5,952 deals (--ga-seeds 24 --deal-seed0 2000), paired with B1's B-v1 /
#      ft-swv games (shipped in data/cache/b1_games), holdout and new-card head-to-heads vs ft-swv, and the principle
#      audit (own-T1 hand-trap fires, opponent-turn interruption rate; ygo-agent's seat as the reference).
# Env: EPOCHS (6), HIST (128), BATCH (512), ANCHOR_DROPS (required), GA_SEEDS (24), VAR_CAP (90m), SMOKE=1 (tiny).
# Usage: bash scripts/vast_stage_b.sh <name>. Do NOT run on the laptop (vast GPU only).
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
EPOCHS=${EPOCHS:-6}; HIST=${HIST:-128}; BATCH=${BATCH:-512}
BV1=data/cache/bv1/best.pt

# 1. shards: the B1 recipe unchanged (vast_stage_b1.sh step 1)
A=data/cache/audit; OLD='data/demos/*,!*/strict-d/*,!*/guides-strict-d/*'
EXCL=("$OLD@$A/strict_junk.json" "data/demos/planned*,!*/strict-d/*@$A/drops-muse.final.jsonl"
      "data/demos/claude/*,!*/strict-d/*@$A/drops-claude.final.jsonl"
      "data/demos/verified/*,data/demos/expanded/*@$A/drops-guides.final.jsonl"
      "*/strict-d/*,*/guides-strict-d/*@$A/drops-strictd.final.jsonl")
[ -s "${ANCHOR_DROPS:-}" ] || { alert "ANCHOR_DROPS missing (${ANCHOR_DROPS:-unset})"; exit 3; }
EXCL+=("data/demos/turns/*@$ANCHOR_DROPS")
for f in $A/strict_junk.json $A/drops-muse.final.jsonl $A/drops-claude.final.jsonl $A/drops-guides.final.jsonl \
         $A/drops-strictd.final.jsonl $BV1; do [ -s "$f" ] || { alert "missing $f"; exit 3; }; done
SONLY=(--only guides claude muse gold teacher dagger sw turns)
[ "${SMOKE:-0}" = 1 ] && SONLY=(--only gold dagger turns)
uv run --no-sync python scripts/build_bc_shards.py "${SONLY[@]}" --strict-labels --out data/cache/bc_shards --jobs "$J" \
  --exclude "${EXCL[@]}" > "$OUT/shards.txt" 2>&1 || { alert "shards failed: $(tail -1 "$OUT/shards.txt")"; exit 3; }
grep -h "^ALERT" "$OUT/shards.txt" >> "$OUT/ALERT"
tick "shards $(grep -m1 '^samples' "$OUT/shards.txt")"
uv run --no-sync python - "$OUT" <<'EOF' || { alert "shard kind check failed: $(tail -1 "$OUT/kinds.txt")"; exit 3; }
import os, sys, collections, numpy as np
from pathlib import Path
c, w = collections.Counter(), collections.defaultdict(float)
for f in sorted(Path("data/cache/bc_shards").glob("*.npz")):
    z = np.load(f)
    if "kind" not in z.files:
        continue
    for k, wt in zip(z["kind"].tolist(), z["weight"].tolist()):
        c[k] += 1
        w[k] += float(wt)
lines = [f"{k}\t{n}\tweighted {w[k]:.0f}" for k, n in sorted(c.items())]
Path(sys.argv[1], "kinds.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
need = ["dagger", "turn_end", "turn_mid", "turn_second"]
need += [] if os.environ.get("SMOKE") == "1" else ["teacher", "line"]
missing = [k for k in need if w.get(k, 0) == 0]
if missing:
    print(f"ALERT 0 weighted samples of {missing}")
    sys.exit(1)
EOF

# 2. B-v3 from scratch: B1's clean kinds and fractions, v3 encoder, long salient history H
HOLD="kewl-tune $( [ -f decks/pool/HOLDOUT ] && tr '\n' ' ' < decks/pool/HOLDOUT )"
K=(teacher teacher_pass dagger dagger_pass line line_improved line_expanded gold turn_mid turn_second turn_end)
F=(line_expanded=0.3 gold=3)
EXTRA=(); [ "${SMOKE:-0}" = 1 ] && EXTRA=(--limit 20000)
# shellcheck disable=SC2086
uv run --no-sync python scripts/train_bc.py --shards data/cache/bc_shards --emb data/cards/emb/qwen3-0.6b-d512.npz \
  --v3 --desc-strings data/cards/emb/strings-qwen3-0.6b-d512.npz --desc-lua data/cards/lua_desc.npz --hist "$HIST" \
  --out "$OUT/v3" --epochs "$EPOCHS" --batch "$BATCH" --lr 5e-4 --holdout $HOLD --select-on last --kinds "${K[@]}" \
  --kind-frac "${F[@]}" "${EXTRA[@]}" > "$OUT/train-v3.txt" 2>&1 &
TP=$!
# epoch metrics -> PROGRESS; after epoch 1 the first milestone (plan-v2 §Stage B), which kills the run on failure. It
# also runs after training if the run finished between two polls (smoke 2026-10-13: 6 epochs in 123 s, never ran).
MS=1; BX=(); [ "${SMOKE:-0}" = 1 ] && { MS=0; BX=(--max-states 300 --min-n 5); }
milestone() {  # exit 0 = passed
  uv run --no-sync python - "$OUT/v3/metrics.jsonl" "$MS" > "$OUT/milestone.txt" 2>&1 <<'EOF2'
import json, math, sys
recs = [json.loads(x) for x in open(sys.argv[1]) if '"epoch"' in x]
ms = int(sys.argv[2])
r = recs[ms]
bad = [] if math.isfinite(r["train_loss"]) and math.isfinite(r["val"]["loss"]) else ["loss not finite"]
t = [x["val"]["per_kind"].get("teacher") for x in recs[: ms + 1]]
if ms > 0 and None in t:
    bad.append(f"no val teacher agreement {t}")
elif ms > 0 and t[-1] <= t[0]:
    bad.append(f"val teacher agreement not rising {t}")
print(f"milestone epoch {ms}: train_loss {r['train_loss']:.4f} val loss {r['val']['loss']:.4f} teacher {t}")
for b in bad:
    print(f"ALERT milestone: {b}")
sys.exit(1 if bad else 0)
EOF2
  local rc=$? rb
  uv run --no-sync python scripts/audit/ctx_band.py --ckpt "$OUT/v3/ep$MS.pt" --out "$OUT/ctx_band-ep$MS.json" \
    "${BX[@]}" > "$OUT/ctx_band-ep$MS.txt" 2>&1; rb=$?
  uv run --no-sync python scripts/audit/ctx_band.py --ckpt "$BV1" --out "$OUT/ctx_band-bv1.json" \
    "${BX[@]}" > "$OUT/ctx_band-bv1.txt" 2>&1
  echo "$(date +%T) milestone $(tail -1 "$OUT/milestone.txt"); band $(grep '^ctx_band' "$OUT/ctx_band-ep$MS.txt"); B-v1 band $(grep '^ctx_band' "$OUT/ctx_band-bv1.txt")" >> "$OUT/PROGRESS"
  grep -h "^ALERT" "$OUT/milestone.txt" "$OUT/ctx_band-ep$MS.txt" >> "$OUT/ALERT"
  grep -q '^ctx_band' "$OUT/ctx_band-ep$MS.txt" || { alert "ctx_band crashed (exit $rb): $(tail -1 "$OUT/ctx_band-ep$MS.txt")"; rb=3; }
  grep -q '^ctx_band' "$OUT/ctx_band-bv1.txt" || alert "ctx_band B-v1 crashed: $(tail -1 "$OUT/ctx_band-bv1.txt")"
  [ $rc -eq 0 ] && [ $rb -eq 0 ]
}
fail_ms() { [ "${SMOKE:-0}" = 1 ] && return; alert "first milestone failed: training stopped"; touch "$OUT/MILESTONE_FAILED"; }
( seen=0; while kill -0 $TP 2>/dev/null; do
    sleep 30; n=$(grep -c '"epoch"' "$OUT/v3/metrics.jsonl" 2>/dev/null); n=${n:-0}
    [ "$n" -gt "$seen" ] || continue
    echo "$(date +%T) train-v3 $(tail -1 "$OUT/v3/metrics.jsonl" | cut -c1-300)" >> "$OUT/PROGRESS"; seen=$n
    if [ "$n" -gt "$MS" ] && [ ! -f "$OUT/milestone.txt" ]; then
      milestone || { fail_ms; [ -f "$OUT/MILESTONE_FAILED" ] && { pkill -TERM -P $TP; kill $TP; }; }
    fi
  done ) &
WP=$!
wait $TP; rc=$?
wait $WP
[ -f "$OUT/milestone.txt" ] || [ $rc -ne 0 ] || milestone || fail_ms
rm -f "$OUT/v3/last.pt"
grep -h "^ALERT" "$OUT/train-v3.txt" >> "$OUT/ALERT"
[ -f "$OUT/MILESTONE_FAILED" ] && exit 4
if [ $rc -ne 0 ] || [ ! -s "$OUT/v3/best.pt" ]; then alert "train-v3 failed (exit $rc): $(tail -1 "$OUT/train-v3.txt")"; exit 3; fi
tick train-v3

# 3. the card on B1's deals; B1's games (B-v1 clean, ft-swv, base_r3) are the paired references
GS=${GA_SEEDS:-24}; [ "${SMOKE:-0}" = 1 ] && GS=1
V="bv3|--opp ygoagent --raw 1 --ckpt $OUT/v3/best.pt"
V="$V;bv3-hold|--opp ckpt --opp-ckpt data/cache/base_r2/best.pt --raw 1 --deck-set holdout --seeds ${HOLD_SEEDS:-2} --ckpt $OUT/v3/best.pt"
V="$V;bv3-new|--opp ckpt --opp-ckpt data/cache/base_r2/best.pt --raw 1 --deck-set newcards --seeds ${NEW_SEEDS:-3} --ckpt $OUT/v3/best.pt"
CKPT=data/cache/base_r3/best.pt SM_ARGS="--ga-seeds $GS --deal-seed0 2000" VARIANTS="$V" PROBE_NPROC=1 \
  VAR_CAP="${VAR_CAP:-90m}" bash scripts/vast_search_match.sh "$NAME"
tick card
for t in bv3 bv3-hold bv3-new; do
  [ -f "$OUT/$t/summary.json" ] && echo "card $t $(tr -d '\n ' < "$OUT/$t/summary.json" | cut -c1-200)" >> "$OUT/PROGRESS"
done
for ref in bclean ftswv base; do
  [ -d "data/cache/b1_games/$ref" ] || continue
  uv run --no-sync python scripts/audit/paired_sm.py "data/cache/b1_games/$ref" "$OUT/bv3" > "$OUT/paired-$ref.txt" 2>&1 \
    || alert "paired_sm vs $ref: $(tail -1 "$OUT/paired-$ref.txt")"
  echo "$(date +%T) paired bv3 - $ref: $(sed -n 2,4p "$OUT/paired-$ref.txt" | tr '\n' ' ')" >> "$OUT/PROGRESS"
done
# gate B 3-4: own-T1 hand-trap fires (games) and the opponent-turn interruption take rate; ygo-agent's seat = reference
for s in s ya; do
  uv run --no-sync python scripts/audit/principles.py --name "bv3-$s" --games "$OUT/bv3" --seat $s \
    --out "$OUT/principles-$s.json" > "$OUT/principles-$s.txt" 2>&1 || alert "principles $s failed: $(tail -1 "$OUT/principles-$s.txt")"
done
uv run --no-sync python - "$OUT" >> "$OUT/PROGRESS" <<'EOF'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
s, y = (json.loads((o / f"principles-{k}.json").read_text()) for k in ("s", "ya"))
first = sum(1 for f in (o / "bv3").glob("games-*.jsonl") for x in f.read_text().splitlines() if x.strip() and json.loads(x)["s_first"])
print(f"gate B: own-T1 hand-trap games {s['lines_with_own_t1_ht']} of {first} going first (cap {33 / 744 * first:.0f}; "
      f"ygo-agent {y['lines_with_own_t1_ht']}); "
      f"interrupt_opp {s['interrupt_opp']['rate']} vs ygo-agent {y['interrupt_opp']['rate']} "
      f"(ratio {s['interrupt_opp']['rate'] / y['interrupt_opp']['rate']:.2f})")
EOF
true
