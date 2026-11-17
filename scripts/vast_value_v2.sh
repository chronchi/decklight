#!/usr/bin/env bash
# Value v2 (docs/lab/prereg/2026-11-11-value-v2.md): value-only RL on Y's frozen trunk (train_rl --value-warmup = --iters:
# only the value head trains, its own optimizer at --value-lr), warm start from Y's league critic (Yv), on Y's league
# rollouts: the 25 cycle-1 specialists + the 10 older Forge (f3) specialists (deck-bound, --deck-opps), self-play, and
# tsoj's bot on EXT_SHARE of each batch. Then gate V1 offline: scripts/audit/leaf_compare.py on held-out games (Y's c1
# card games, not trained on): v2 vs value_p2 (Yv reported). V2/H1/H2 run in later jobs/on the laptop.
# Stages: S (3 iterations, invariants, else stop), T (VITERS), export (policy must equal Y; head must differ from Yv's),
# V1. Env: V2_IN (an exported v2 value best.pt: V1 only), VITERS (120), EXT_SHARE (0.2), TRAIN_SEEDS (300), CAP_T (3.5h), RL_HOURS (3.2), LRES (10), V1_GAMES (3000).
# Usage: bash scripts/vast_value_v2.sh <name>
set -uo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
NAME="$1"; OUT="$PWD/runs/tsoj-arms/$NAME"; mkdir -p "$OUT" "runs/$NAME"
trap 'rsync -a --exclude "*.stream.jsonl.gz" "$OUT"/ "runs/$NAME"/ 2>/dev/null || cp -r "$OUT"/. "runs/$NAME"/; pkill -P $$ 2>/dev/null' EXIT
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
progress() { echo "$(date +%T) $*" | tee -a "$OUT/PROGRESS"; }
export OMP_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=data/cache/value_v2  # staged by scripts/launch/launch-value-v2.sh
INIT=$G/Y/best.pt; VINIT=$G/Yv/best.pt; VP2=$G/value_p2/best.pt; MANI=$G/opps/manifest.json
for f in $INIT $G/Y/metrics.jsonl $VINIT $G/Yv/metrics.jsonl $VP2 $G/value_p2/metrics.jsonl $MANI; do
  [ -s "$f" ] || { alert "missing $f"; exit 3; }
done
NV=$(cat $G/v1games/*.jsonl 2>/dev/null | wc -l); [ "$NV" -ge 1000 ] || { alert "only $NV V1 games staged"; exit 3; }
find bots -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null
uv sync --group dev --group ml -q --no-install-package torch || { alert "uv sync failed"; exit 3; }
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
. scripts/cores.sh
J=$(box_cores)
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &

progress "toolchain"
command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null || { alert "rustup failed"; exit 3; }; }
command -v cmake >/dev/null || { uv tool install -q cmake || { alert "cmake install failed"; exit 3; }; }
export YGO_AI="$HOME/yugioh_ai"
[ -d "$YGO_AI" ] || git clone -q https://github.com/tsoj/yugioh_ai "$YGO_AI" || { alert "clone failed"; exit 3; }
READY_TIMEOUT=300 TASKSET="" CARGO_BUILD_JOBS="$J" bash scripts/tsoj/build_tsoj.sh > "$OUT/build.log" 2>&1 || { alert "build_tsoj failed: $(tail -3 "$OUT/build.log" | tr '\n' ' ')"; exit 3; }
progress "build done"

B=scripts/tsoj/books
uv run --no-sync python scripts/tsoj/make_book.py --mirror --seeds "${TRAIN_SEEDS:-300}" --seed-base 100000 --out $B/train.jsonl | head -1 | tee -a "$OUT/PROGRESS"
uv run --no-sync python scripts/tsoj/make_book.py --mirror --seeds "${EVAL_SEEDS:-28}" --seed-base 20000 --out $B/heldout.jsonl | head -1 | tee -a "$OUT/PROGRESS"
NA=$(( (J - ${LRES:-10}) * ${ACT_PER_CORE:-2} )); [ "$NA" -lt 8 ] && { alert "only $J cores: $NA actor slots"; exit 3; }
echo "cores $J -> $NA actor games at once, ${LRES:-10} cores for the learner + server" | tee "$OUT/sizing.txt"
SOCK=/tmp/dlp-$$.sock

start_ext() {  # start_ext DIR: policy server on DIR/weights.pt (version 0 = X2) + actor loop writing episodes to DIR
  local d="$1"; mkdir -p "$d"; rm -f "$d/ready" "$d/STOP"
  PYTHONUNBUFFERED=1 uv run --no-sync python -u scripts/tsoj/policy_server.py --sock "$SOCK" --weights "$d/weights.pt" \
    --init $INIT --value-init $VINIT --device cuda --ready "$d/ready" > "$d/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 120); do [ -f "$d/ready" ] && break; sleep 1; done
  [ -f "$d/ready" ] || { alert "policy server not ready: $(tail -3 "$d/server.log" | tr '\n' ' ')"; return 1; }
  ( k=0; nb=$(wc -l < $B/train.jsonl); empty=0; stage=$(basename "$(dirname "$d")")
    while [ ! -f "$d/STOP" ]; do
      lo=$(( (k * NA) % nb + 1 )); sed -n "${lo},$(( lo + NA - 1 ))p" $B/train.jsonl > "$d/chunk.jsonl"
      e0=$(find "$d" -name "ep-*.npz" | wc -l)
      DL_AGENT=remote DL_POLICY_SOCK="$SOCK" DL_EPISODE_DIR="$d" OUTNAME="ext-$stage-$k" \
        timeout 40m bash scripts/tsoj/run_arms.sh "$NAME" p 500 "$NA" "$NA" "$d/chunk.jsonl" >> "$d/actors.log" 2>&1 \
        || echo "ALERT actors chunk $k exit $?" >> "$d/actors.log"
      [ -f "$d/STOP" ] && break
      g=$(grep -c . "$OUT/ext-$stage-$k.jsonl" 2>/dev/null); g=${g:-0}
      e1=$(find "$d" -name "ep-*.npz" | wc -l)
      echo "chunk $k: $g games, $(( e1 - e0 )) new episodes, $(grep -c 'did not answer' "$OUT/ext-$stage-$k.log" 2>/dev/null) isready timeouts $(date +%T)" >> "$d/chunks.txt"
      if [ "$g" -gt 0 ] && [ "$e1" -le "$e0" ]; then
        alert "ext $stage chunk $k: $g games but 0 new episodes in $d"; pkill -f "scripts/train_rl.py"; break
      fi
      [ "$g" -eq 0 ] && empty=$(( empty + 1 )) || empty=0
      if [ "$empty" -ge 2 ]; then
        alert "ext $stage: 2 chunks with 0 games: $(grep -m1 error "$OUT/ext-$stage-$k.log" | cut -c1-200)"; pkill -f "scripts/train_rl.py"; break
      fi
      k=$(( k + 1 ))
    done ) &
  ACTORS_PID=$!
}
stop_ext() {
  touch "$1/STOP"; pkill -f "ygo-arbiter" 2>/dev/null; sleep 5
  kill "$ACTORS_PID" "$SERVER_PID" 2>/dev/null; wait "$ACTORS_PID" "$SERVER_PID" 2>/dev/null
  grep -h "^ALERT" "$1/actors.log" "$1/server.log" 2>/dev/null | sort | uniq -c | sed 's/^/ALERT ext: /' >> "$OUT/ALERT"
}
COMMON="--init $INIT --value-init $VINIT --value-lr ${VLR:-1e-4} --lr 4e-5 --kl-coef 0.03
  --eval-every 1000000000 --device cuda --no-ygoagent --league 1 --p-self 0.4 --p-ygoagent 0 --deck-opps $MANI
  --threads 4 --hours ${RL_HOURS:-3.2} --save-every 20"
EXT="--ext-share ${EXT_SHARE:-0.2}"
train() {  # train OUT ITERS [args...]: value only (every iteration a warm-up); TCAP = time cap (0 = none)
  local o="$1" n="$2"; shift 2
  # shellcheck disable=SC2086
  PYTHONUNBUFFERED=1 timeout "${TCAP:-0}" uv run --no-sync python -u scripts/train_rl.py $COMMON --iters "$n" --value-warmup "$n" \
    --out "$o" "$@" > "$o.log" 2>&1
}
metric() {  # one line per iteration: iter, vf, games
  python3 - "$1" <<'PY'
import json, sys
for x in open(sys.argv[1]):
    r = json.loads(x)
    if "iter" in r and "samples" in r:
        print(r["iter"], {k: v for k, v in r.items() if k.startswith(("ext_", "deck_opp", "vf", "value")) or k == "samples"})
PY
}
check_frozen() {  # check_frozen RL_CKPT: the policy (every non-value_head tensor) must equal Y's
  uv run --no-sync python - "$1" "$INIT" <<'PY'
import sys, torch
a, b = (torch.load(p, map_location="cpu") for p in sys.argv[1:3])
a, b = a.get("model", a), b.get("model", b)
moved = [k for k in b if not k.startswith("value_head.") and k in a and not torch.equal(a[k].float(), b[k].float())]
missing = [k for k in b if not k.startswith("value_head.") and k not in a]
print(f"policy tensors moved {len(moved)}, missing {len(missing)}: {moved[:3]} {missing[:3]}")
sys.exit(1 if moved or missing else 0)
PY
}

if [ -n "${V2_IN:-}" ]; then  # V1 only: an exported v2 value net (S, T and export skipped)
  V2=$V2_IN; [ -s "$V2" ] || { alert "V2_IN $V2 missing"; exit 4; }; progress "V1 only: $V2"
else
# ---- S: smoke ----
S="$OUT/smoke"; mkdir -p "$S"
start_ext "$S/ext" || exit 4
train "$S/V" 3 --ext-dir "$S/ext" $EXT --ext-wait-s 900; RV=$?
stop_ext "$S/ext"
progress "smoke V exit $RV: $(metric "$S/V/metrics.jsonl" 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-900)"
ok=1
[ "$RV" = 0 ] || { alert "smoke train_rl exit $RV: $(grep -h 'ALERT\|Error' "$S/V.log" | tail -3 | tr '\n' ' ')"; ok=0; }
python3 - "$S/V/metrics.jsonl" "$S/V.log" <<'PY' || ok=0
import json, math, re, sys
rec = [r for r in (json.loads(x) for x in open(sys.argv[1]) if '"iter"' in x) if r["iter"] > 0 and "samples" in r]
bad = []
if len(rec) < 3: bad.append(f"wrote {len(rec)} iterations")
for r in rec:
    it = r["iter"]
    if r.get("ext_games", 0) == 0: bad.append(f"it {it}: 0 tsoj games")
    if r.get("deck_opp_games", 0) == 0: bad.append(f"it {it}: 0 deck-bound specialist games")
    vf = [v for k, v in r.items() if k.startswith("vf")]
    if not vf or any(not math.isfinite(v) for v in vf): bad.append(f"it {it}: vf missing or not finite {vf}")
m = re.search(r"\[deck-opps\] (\d+) nets", open(sys.argv[2]).read())
if not m or int(m.group(1)) < 30: bad.append(f"deck-opps nets: {m.group(1) if m else 'none'} (want 35)")
for b in bad: print("ALERT smoke:", b)
sys.exit(1 if bad else 0)
PY
[ "$ok" = 1 ] || { alert "smoke failed: stop before training"; exit 4; }
CKS=$(ls "$S"/V/last.pt "$S"/V/snap-*.pt 2>/dev/null | head -1)
[ -n "$CKS" ] && { check_frozen "$CKS" >> "$OUT/PROGRESS" 2>&1 || { alert "smoke: policy weights moved"; exit 4; }; }
progress "smoke OK"

# ---- T: value head training ----
T="$OUT/train"; mkdir -p "$T"
start_ext "$T/ext" || exit 4
TCAP="${CAP_T:-3.5h}" train "$T/V" "${VITERS:-120}" --ext-dir "$T/ext" $EXT &
TV=$!
( while sleep 300; do [ -f "$T/V/metrics.jsonl" ] && progress "PROGRESS T V $(metric "$T/V/metrics.jsonl" | tail -1 | cut -c1-500)"; done ) &
MON=$!
wait $TV; RV=$?
progress "T V exit $RV"
stop_ext "$T/ext"
kill $MON 2>/dev/null
[ "$RV" = 0 ] || [ "$RV" = 124 ] || { alert "T V failed ($RV): $(grep -h 'ALERT\|Error' "$T/V.log" | tail -3 | tr '\n' ' ')"; exit 4; }
CK=""; if [ ! -s "$T/V/last.pt" ]; then
  CK=$(ls "$T"/V/snap-*.pt 2>/dev/null | sort | tail -1); [ -n "$CK" ] || { alert "no last.pt and no snapshot"; exit 4; }
  alert "no last.pt (time cap?): exporting $CK"; CK="--ckpt $CK"
fi
check_frozen "$( [ -n "$CK" ] && echo "${CK#--ckpt }" || echo "$T/V/last.pt")" >> "$OUT/PROGRESS" 2>&1 || { alert "policy weights moved in training"; exit 4; }
# shellcheck disable=SC2086
uv run --no-sync python scripts/tsoj/export_rl.py "$T/V" "$T/export-V" --init $INIT --value-init $VINIT --value-only $CK > "$T/export-V.log" 2>&1 \
  || { alert "export failed: $(tail -2 "$T/export-V.log" | tr '\n' ' ')"; exit 4; }
V2="$T/export-V/value/best.pt"
cmp -s "$V2" $VINIT && { alert "v2 is Yv (head did not move)"; exit 4; }
progress "export: $V2"
fi

# ---- V1: offline winner prediction on held-out games (v2 vs value_p2; Yv reported) ----
uv run --no-sync python scripts/audit/leaf_compare.py --games $G/v1games --value "v2=$V2" --value "p2=$VP2" --value "yv=$VINIT" \
  --max-games "${V1_GAMES:-3000}" --device cuda --deck-dir $G/decks --out "$OUT/v1.json" > "$OUT/v1.txt" 2>&1 \
  || { alert "V1 leaf_compare failed: $(tail -3 "$OUT/v1.txt" | tr '\n' ' ')"; exit 4; }
progress "V1: $(tail -2 "$OUT/v1.txt" | tr '\n' ' ')"
python3 - "$OUT/v1.json" <<'PY' | tee -a "$OUT/PROGRESS"
import json, sys
d = json.load(open(sys.argv[1]))["turn_start_dmse"]
print(f"GATE V1 {'PASS' if d['dmse'] < 0 else 'FAIL'}: turn-start dMSE v2 - p2 = {d['dmse']:+.5f} (SE {d['se']:.5f}); "
      "Brier = MSE/4, so the sign is the Brier difference's")
PY
progress "done"
