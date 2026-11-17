#!/usr/bin/env bash
# Cycle-1 league segment (docs/lab/prereg/2026-11-10-master-cycle1.md, amendments 6-7): PPO from X2 (observable history)
# against the 25 accepted c1 specialists (deck-bound, --deck-opps) and the self-play league, plus tsoj's bot on
# EXT_SHARE of each batch (--ext-dir); KL to X2. One GPU box.
# Stages: S (3 iterations, every invariant checked, else stop), T (ITERS; fail fast at it40 on the tsoj win rate),
# E (held-out tsoj gauntlet, 1,568 games, paired vs X's G4 re-play), C (the c1 card's N rows for Y, against the R rows
# of the round's card copied in: retention on the 25 archetypes, old, reserve, gauntlet; plus the mirror no-loss card
# Y vs X), then scripts/audit/c1_league_gates.py prints the gates.
# Env: ITERS (80), KL (0.03), EXT_SHARE (0.2), TRAIN_SEEDS (300), EVAL_SEEDS (28), CAP_T (4h), CAP_E (70m),
# RL_HOURS (3.4), LRES (10), VAR_CAP (40m), NEW_GAMES (1536), OLD_GAMES (1536), RES_GAMES (3072), GA_SEEDS (24),
# MIRROR_GAMES (3072), Y_IN (an exported Y policy: skip S and T). Usage: bash scripts/vast_c1_league.sh <name>
set -uo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
NAME="$1"; OUT="$PWD/runs/tsoj-arms/$NAME"; mkdir -p "$OUT" "runs/$NAME"
trap 'rsync -a --exclude "*.stream.jsonl.gz" "$OUT"/ "runs/$NAME"/ 2>/dev/null || cp -r "$OUT"/. "runs/$NAME"/; pkill -P $$ 2>/dev/null' EXIT
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
progress() { echo "$(date +%T) $*" | tee -a "$OUT/PROGRESS"; }
export OMP_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=data/cache/c1_league  # staged by scripts/launch/launch-c1-league.sh
INIT=$G/X2/best.pt; VINIT=$G/Xv/best.pt; XB=$G/X/best.pt; MANI=$G/opps/manifest.json
for f in $INIT $G/X2/metrics.jsonl $VINIT $G/Xv/metrics.jsonl $XB $G/X/metrics.jsonl $MANI $G/eval-X.jsonl \
         $G/eval-X2.jsonl $G/verify.json $G/mirror.pairs; do
  [ -s "$f" ] || { alert "missing $f"; exit 3; }
done
cmp -s $INIT $XB && { alert "X2 and X are the same file"; exit 3; }
NR=$(ls -d $G/cardR/R-* 2>/dev/null | wc -l); [ "$NR" = 41 ] || { alert "expected 41 R rows in $G/cardR, got $NR"; exit 3; }
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
COMMON="--init $INIT --value-init $VINIT --lr 4e-5 --kl-coef ${KL:-0.03}
  --eval-every 1000000000 --device cuda --no-ygoagent --league 1 --p-self 0.4 --p-ygoagent 0 --deck-opps $MANI
  --threads 4 --hours ${RL_HOURS:-3.4} --save-every 20"
EXT="--ext-share ${EXT_SHARE:-0.2}"
train() {  # train OUT ITERS [args...]; TCAP = time cap (0 = none)
  local o="$1" n="$2"; shift 2
  # shellcheck disable=SC2086
  PYTHONUNBUFFERED=1 timeout "${TCAP:-0}" uv run --no-sync python -u scripts/train_rl.py $COMMON --iters "$n" --out "$o" "$@" \
    > "$o.log" 2>&1
}
metric() {  # metric METRICS.jsonl: one line per iteration (iter, ext + league fields)
  python3 - "$1" <<'EOF'
import json, sys
for x in open(sys.argv[1]):
    r = json.loads(x)
    if "iter" in r:
        print(r["iter"], {k: v for k, v in r.items() if k.startswith(("ext_", "league", "deck_opp")) or k in ("samples", "kl_bc", "train_vs_bc_sampled")})
EOF
}

if [ -z "${Y_IN:-}" ]; then  # Y_IN: an exported Y policy (staged) -> skip S and T, go straight to E and C
# ---- S: smoke ----
S="$OUT/smoke"; mkdir -p "$S"
start_ext "$S/ext" || exit 4
train "$S/Y" 3 --ext-dir "$S/ext" $EXT --ext-wait-s 900; RY=$?
stop_ext "$S/ext"
progress "smoke ext chunks: $(tr '\n' ';' < "$S/ext/chunks.txt" 2>/dev/null | cut -c1-500)"
progress "smoke Y exit $RY: $(metric "$S/Y/metrics.jsonl" 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-900)"
ok=1
[ "$RY" = 0 ] || { alert "smoke train_rl exit $RY: $(grep -h 'ALERT\|Error' "$S/Y.log" | tail -3 | tr '\n' ' ')"; ok=0; }
python3 - "$S/Y/metrics.jsonl" "$S/Y.log" <<'EOF' || ok=0
import json, re, sys
rec = [r for r in (json.loads(x) for x in open(sys.argv[1]) if '"iter"' in x) if r["iter"] > 0 and "samples" in r]
bad = []
if len(rec) < 3: bad.append(f"Y wrote {len(rec)} iterations")
for r in rec:
    it = r["iter"]
    if r.get("ext_games", 0) == 0: bad.append(f"it {it}: 0 tsoj games")
    if r.get("deck_opp_games", 0) == 0: bad.append(f"it {it}: 0 deck-bound specialist games")
    if r.get("kl_bc", 0) != r.get("kl_bc", 0): bad.append(f"it {it}: kl_bc NaN")
print("ext/deck_opp:", [(r.get("ext_games"), r.get("ext_wr"), r.get("deck_opp_games"), r.get("deck_opp_wr")) for r in rec])
if not re.search(r"\[deck-opps\] 25 nets, 75 bound decks", open(sys.argv[2]).read()):
    bad.append("deck-opps not loaded as 25 nets / 75 bound decks")
for b in bad: print("ALERT smoke:", b)
sys.exit(1 if bad else 0)
EOF
[ "$ok" = 1 ] || { alert "smoke failed: stop before training"; exit 4; }
progress "smoke OK"

# ---- T: train Y; it40 fail-fast on the tsoj win rate (prereg amendment 7) ----
T="$OUT/train"; mkdir -p "$T"
start_ext "$T/ext" || exit 4
TCAP="${CAP_T:-4h}" train "$T/Y" "${ITERS:-80}" --ext-dir "$T/ext" $EXT &
TY=$!
( while sleep 300; do [ -f "$T/Y/metrics.jsonl" ] && progress "PROGRESS T Y $(metric "$T/Y/metrics.jsonl" | tail -1 | cut -c1-700)"; done ) &
MON=$!
( while :; do
    n=$(python3 -c "import json,sys; print(max([json.loads(x).get('iter',0) for x in open(sys.argv[1]) if '\"samples\"' in x] or [0]))" "$T/Y/metrics.jsonl" 2>/dev/null)
    [ "${n:-0}" -ge 40 ] && break
    kill -0 $TY 2>/dev/null || exit 0; sleep 60
  done
  python3 scripts/audit/c1_league_gates.py failfast "$T/Y/metrics.jsonl" > "$T/failfast.txt" 2>&1; r=$?
  progress "it40 fail-fast: $(tail -1 "$T/failfast.txt")"
  [ $r = 0 ] || { alert "it40 fail-fast: $(tail -1 "$T/failfast.txt"); stopping training"; pkill -f "scripts/train_rl.py"; touch "$T/FAILFAST"; }
) &
MS=$!
wait $TY; RY=$?
progress "T Y exit $RY"
stop_ext "$T/ext"
kill $MON 2>/dev/null
wait $MS 2>/dev/null
[ -f "$T/FAILFAST" ] && { progress "stopped at it40 by the pre-registered fail-fast"; exit 5; }
[ "$RY" = 0 ] || [ "$RY" = 124 ] || { alert "T Y failed ($RY): $(grep -h 'ALERT\|Error' "$T/Y.log" | tail -3 | tr '\n' ' ')"; exit 4; }
CK=""; if [ ! -s "$T/Y/last.pt" ]; then
  CK=$(ls "$T"/Y/snap-*.pt 2>/dev/null | sort | tail -1); [ -n "$CK" ] || { alert "no last.pt and no snapshot"; exit 4; }
  alert "no last.pt (time cap?): exporting $CK"; CK="--ckpt $CK"
fi
# shellcheck disable=SC2086
uv run --no-sync python scripts/tsoj/export_rl.py "$T/Y" "$T/export-Y" --init $INIT --value-init $VINIT $CK > "$T/export-Y.log" 2>&1 \
  || { alert "export Y failed: $(tail -2 "$T/export-Y.log" | tr '\n' ' ')"; exit 4; }
YB="$T/export-Y/policy/best.pt"
cmp -s "$YB" $INIT && { alert "Y is X2"; exit 4; }
else
  YB="$Y_IN"; [ -s "$YB" ] && [ -s "$(dirname "$YB")/metrics.jsonl" ] || { alert "missing $YB or its metrics.jsonl"; exit 3; }
  cmp -s "$YB" $INIT && { alert "Y is X2"; exit 4; }
  progress "resume: Y = $YB (training done in an earlier job)"
fi

# ---- E: held-out tsoj gauntlet for Y, paired with X's and X2's G4 re-plays ----
t0=$(date +%s); f0=$(date +%Y-%m-%dT%H:%M:%S)
DL_AGENT=raw DL_CKPT="$PWD/$YB" OUTNAME="eval-Y" timeout "${CAP_E:-70m}" bash scripts/tsoj/run_arms.sh "$NAME" p 500 \
  "$(wc -l < $B/heldout.jsonl)" "$(( J - 2 ))" $B/heldout.jsonl > "$OUT/run-eval-Y.txt" 2>&1 || alert "eval Y exit $?"
f1=$(date +%Y-%m-%dT%H:%M:%S)
uv run --no-sync python scripts/tsoj/gauntlet_summary.py "Y=$OUT/eval-Y.jsonl:$(( $(date +%s) - t0 ))" "X=$G/eval-X.jsonl" \
  "X2=$G/eval-X2.jsonl" --paired X --out "$OUT/eval.json" > "$OUT/eval.txt" 2>&1 || alert "eval summary failed: $(tail -2 "$OUT/eval.txt" | tr '\n' ' ')"
progress "E: $(grep -h '"Y"\|paired' "$OUT/eval.txt" | cut -c1-900 | tr '\n' ' ')"
python3 scripts/tsoj/turn2_interrupts.py "Y=$OUT/logs@$f0,$f1" --out "$OUT/turn2.json" > "$OUT/turn2.txt" 2>&1 || alert "turn2 audit failed: $(tail -2 "$OUT/turn2.txt" | tr '\n' ' ')"
progress "turn2: $(cut -c1-600 "$OUT/turn2.txt" | tr '\n' ' ')"

# ---- C: the c1 card's N rows for Y (R rows = the round's X rows, same deals) + the mirror no-loss card ----
C="runs/$NAME/card"; mkdir -p "$C"
cp -r $G/cardR/R-* "$C"/
N="--ckpt $PWD/$YB"; V=""
for a in $(python3 -c "import json,sys; print(' '.join(k for k, v in json.load(open(sys.argv[1])).items() if v['decision'] == 'ACCEPT'))" $G/verify.json); do
  s=$(uv run --no-sync python scripts/forge_round_prep.py seeds --pairs "$G/vpairs/$a.pairs" --games "${NEW_GAMES:-1536}")
  V="$V;N-$a|$N --pairs @$G/vpairs/$a.pairs --seeds $s"
done
OLD_ARCHS=$(cat $G/old_archs.txt)
for a in $OLD_ARCHS; do
  s=$(uv run --no-sync python scripts/forge_round_prep.py seeds --pairs "decks/forge/c1/old-pairs/arch-$a.pairs" --games "${OLD_GAMES:-1536}")
  V="$V;N-$a|$N --pairs @decks/forge/c1/old-pairs/arch-$a.pairs --seeds $s"
done
s=$(uv run --no-sync python scripts/forge_round_prep.py seeds --pairs decks/forge/f2/pairs/reserve.pairs --games "${RES_GAMES:-3072}")
V="$V;N-res|$N --pairs @decks/forge/f2/pairs/reserve.pairs --seeds $s"
V="$V;N-ga|$N --opp ygoagent --ga-seeds ${GA_SEEDS:-24}"
s=$(uv run --no-sync python scripts/forge_round_prep.py seeds --pairs $G/mirror.pairs --games "${MIRROR_GAMES:-3072}")
V="$V;M-mirror|$N --pairs @$G/mirror.pairs --seeds $s"
V="${V#;}"
SM="--raw 1 --opp ckpt --opp-ckpt $PWD/$XB --seed-per-pair 1 --deal-seed0 43000 --deck-dir $G/card-decks"
progress "card variants: $(echo "$V" | tr ';' '\n' | cut -d'|' -f1 | tr '\n' ' ')"
CKPT=$XB SM_ARGS="$SM" VARIANTS="$V" VAR_CAP="${VAR_CAP:-40m}" PROBE_NPROC=40 \
  bash scripts/vast_search_match.sh "$NAME/card" > "$OUT/card.log" 2>&1 || alert "card job exit $?"
# shellcheck disable=SC2086
uv run --no-sync python scripts/audit/forge_release_card.py "$C" --verify $G/verify.json \
  --new $(python3 -c "import json,sys; print(' '.join(k for k, v in json.load(open(sys.argv[1])).items() if v['decision'] == 'ACCEPT'))" $G/verify.json) \
  --old $OLD_ARCHS --name Y > "$OUT/release_card.txt" 2>&1 || alert "forge_release_card failed: $(tail -1 "$OUT/release_card.txt")"
progress "card: $(grep -h '^GATE\|^RELEASE' "$OUT/release_card.txt" | tr '\n' ' ')"
uv run --no-sync python scripts/audit/c1_league_gates.py gates --eval "$OUT/eval.json" --card "runs/$NAME/release_card.json" \
  --mirror "$C/M-mirror" --out "$OUT/gates.json" > "$OUT/gates.txt" 2>&1 || alert "gates: $(tail -1 "$OUT/gates.txt")"
progress "GATES: $(tr '\n' ' ' < "$OUT/gates.txt" | cut -c1-900)"
exit 0
