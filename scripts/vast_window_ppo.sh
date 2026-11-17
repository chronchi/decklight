#!/usr/bin/env bash
# Window PPO (NOTEBOOK 2026-11-08 window PPO pre-registration; user: "try lower KL to allow more exploration and maybe
# we can have a pool of decks/opponents that really kill in one turn ... or combo a lot"): one arm X from A (tsoj PPO
# arm A, export-A) on one GPU box.
#   X: league (self-play, frozen snapshots; no specialists: they cannot be combined with --line-starts) + heavy combo
#      lines (starts-v2-heavy, DAGS_P of episodes) + tsoj's bot (EXT_SHARE), KL to A at KL (0.03) and 0 on opponent-turn
#      chain-0 windows (--kl-win-mult 0), forced activations there in combo-line episodes only (--win-explore 0.25
#      --isw-floor 0.5 --win-explore-lines-only 1).
# Stages: S (3 iterations, every invariant checked, else stop), T (ITERS; at it20/it40 the window gate on the 614
# stored held-out windows, no search: --reuse; fail fast at it40), E (held-out tsoj gauntlet, the same openings as the
# stored eval-A/eval-P of tsoj-ppo-20261107-2123, paired; the turn-2 interruption audit).
# Env: ITERS (80), KL (0.03), DAGS_P (0.5), EXT_SHARE (0.2), TRAIN_SEEDS (300), EVAL_SEEDS (28), CAP_T (4h), CAP_E
# (70m), RL_HOURS (3.3: train_rl stops itself and writes last.pt; CAP_T is the backstop), LRES (10). Usage: bash scripts/vast_window_ppo.sh <name>
set -uo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
NAME="$1"; OUT="$PWD/runs/tsoj-arms/$NAME"; mkdir -p "$OUT" "runs/$NAME"
trap 'rsync -a --exclude "*.stream.jsonl.gz" "$OUT"/ "runs/$NAME"/ 2>/dev/null || cp -r "$OUT"/. "runs/$NAME"/; pkill -P $$ 2>/dev/null' EXIT
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
progress() { echo "$*" | tee -a "$OUT/PROGRESS"; }
export OMP_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=data/cache/window_ppo  # init (A's export), stored evals and window rows, shipped by the launcher
INIT=$W/A/policy/best.pt; VINIT=$W/A/value/best.pt; LS=data/cache/line_starts/starts-v2-heavy.jsonl
for f in $INIT $VINIT $W/A/policy/metrics.jsonl $W/eval-A.jsonl $W/eval-P.jsonl $LS data/cache/line_starts/starts-v2.eval.jsonl; do
  [ -s "$f" ] || { alert "missing $f"; exit 3; }
done
[ "$(ls $W/wg/s[0-9]*.json | wc -l)" = 30 ] || { alert "expected 30 stored window shards in $W/wg"; exit 3; }
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

echo "== toolchain $(date +%T)"
command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null || { alert "rustup failed"; exit 3; }; }
command -v cmake >/dev/null || { uv tool install -q cmake || { alert "cmake install failed"; exit 3; }; }
export YGO_AI="$HOME/yugioh_ai"
[ -d "$YGO_AI" ] || git clone -q https://github.com/tsoj/yugioh_ai "$YGO_AI" || { alert "clone failed"; exit 3; }
READY_TIMEOUT=300 TASKSET="" CARGO_BUILD_JOBS="$J" bash scripts/tsoj/build_tsoj.sh > "$OUT/build.log" 2>&1 || { alert "build_tsoj failed: $(tail -3 "$OUT/build.log" | tr '\n' ' ')"; exit 3; }
progress "build done $(date +%T)"

B=scripts/tsoj/books
uv run --no-sync python scripts/tsoj/make_book.py --mirror --seeds "${TRAIN_SEEDS:-300}" --seed-base 100000 --out $B/train.jsonl | head -1 | tee -a "$OUT/PROGRESS"
uv run --no-sync python scripts/tsoj/make_book.py --mirror --seeds "${EVAL_SEEDS:-28}" --seed-base 20000 --out $B/heldout.jsonl | head -1 | tee -a "$OUT/PROGRESS"
NA=$(( (J - ${LRES:-10}) * ${ACT_PER_CORE:-2} )); [ "$NA" -lt 8 ] && { alert "only $J cores: $NA actor slots"; exit 3; }
echo "cores $J -> $NA actor games at once, ${LRES:-10} cores for the learner + server" | tee "$OUT/sizing.txt"
SOCK=/tmp/dlp-$$.sock

start_ext() {  # start_ext DIR: policy server on DIR/weights.pt (version 0 = A) + actor loop writing episodes to DIR
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
COMMON="--init $INIT --value-init $VINIT --lr 4e-5 --kl-coef ${KL:-0.03} --kl-win-mult 0
  --win-explore 0.25 --isw-floor 0.5 --win-explore-lines-only 1 --line-starts $LS --dags-p ${DAGS_P:-0.5}
  --eval-every 1000000000 --device cuda --no-ygoagent --league 1 --p-self 0.4 --p-ygoagent 0
  --threads 4 --hours ${RL_HOURS:-3.3} --save-every 20"
train() {  # train OUT ITERS [ext args...]; TCAP = time cap (0 = none)
  local o="$1" n="$2"; shift 2
  # shellcheck disable=SC2086
  PYTHONUNBUFFERED=1 timeout "${TCAP:-0}" uv run --no-sync python -u scripts/train_rl.py $COMMON --iters "$n" --out "$o" "$@" \
    > "$o.log" 2>&1
}
metric() {
  python3 - "$1" <<'EOF'
import json, sys
for x in open(sys.argv[1]):
    r = json.loads(x)
    if "iter" in r:
        print(r["iter"], {k: v for k, v in r.items() if k.startswith(("ext_", "line_", "win_", "isw")) or k in ("samples", "kl_bc", "train_vs_bc_sampled")})
EOF
}
EXT="--ext-share ${EXT_SHARE:-0.2}"

# ---- S: smoke ----
S="$OUT/smoke"; mkdir -p "$S"
start_ext "$S/ext" || exit 4
train "$S/X" 3 --ext-dir "$S/ext" $EXT --ext-wait-s 900; RX=$?
stop_ext "$S/ext"
progress "smoke ext chunks: $(tr '\n' ';' < "$S/ext/chunks.txt" 2>/dev/null | cut -c1-500)"
progress "smoke X exit $RX: $(metric "$S/X/metrics.jsonl" 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-900)"
ok=1
[ "$RX" = 0 ] || { alert "smoke train_rl exit $RX: $(grep -h 'ALERT\|Error' "$S/X.log" | tail -3 | tr '\n' ' ')"; ok=0; }
python3 - "$S/X/metrics.jsonl" <<'EOF' || ok=0
import json, sys
rec = [r for r in (json.loads(x) for x in open(sys.argv[1]) if '"iter"' in x) if r["iter"] > 0 and "samples" in r]
bad = []
if len(rec) < 3: bad.append(f"X wrote {len(rec)} iterations")
for r in rec:
    it = r["iter"]
    if r.get("ext_games", 0) == 0: bad.append(f"it {it}: 0 tsoj games")
    if r.get("line_games", 0) == 0: bad.append(f"it {it}: 0 combo-line games")
    if r.get("win_rows", 0) == 0: bad.append(f"it {it}: 0 explored window rows")
    if r.get("win_explored", 0) == 0: bad.append(f"it {it}: 0 forced activations")
    if r.get("kl_bc", 0) != r.get("kl_bc", 0): bad.append(f"it {it}: kl_bc NaN")
print("line/win:", [(r.get("line_games"), r.get("line_wr"), r.get("win_rows"), r.get("win_explored"), r.get("isw_floored_share")) for r in rec])
for b in bad: print("ALERT smoke:", b)
sys.exit(1 if bad else 0)
EOF
[ "$ok" = 1 ] || { alert "smoke failed: stop before training"; exit 4; }
progress "smoke OK $(date +%T)"

# ---- T: train X, window gate at the milestones ----
T="$OUT/train"; mkdir -p "$T"
wg() {  # wg IT: export snap-IT, then the window gate on the stored windows (CPU, no search)
  local it; it=$(printf %04d "$1")
  uv run --no-sync python scripts/tsoj/export_rl.py "$T/X" "$T/export-$it" --init $INIT --value-init $VINIT \
    --ckpt "$T/X/snap-$it.pt" > "$T/export-$it.log" 2>&1 || { alert "export snap-$it failed: $(tail -2 "$T/export-$it.log" | tr '\n' ' ')"; return 1; }
  # --cand must differ from the product P (forge_r1): a P-identical X is caught here as exit 2
  OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 uv run --no-sync python -u scripts/audit/window_gate.py --starts data/cache/line_starts/starts-v2.eval.jsonl \
    --cand "$T/export-$it/policy/best.pt" --reuse $W/wg/s[0-9]*.json --out "$T/wg-$it.json" > "$T/wg-$it.txt" 2>&1
  local r=$?; [ "$r" = 0 ] || { alert "window gate it$1 exit $r: $(tail -2 "$T/wg-$it.txt" | tr '\n' ' ')"; return 1; }
  progress "WG it$1: $(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['summary']))" "$T/wg-$it.json")"
}
start_ext "$T/ext" || exit 4
TCAP="${CAP_T:-4h}" train "$T/X" "${ITERS:-80}" --ext-dir "$T/ext" $EXT &
TX=$!
( while sleep 300; do [ -f "$T/X/metrics.jsonl" ] && progress "T X $(metric "$T/X/metrics.jsonl" | tail -1 | cut -c1-700)"; done ) &
MON=$!
# milestone watcher: it20 informs, it40 is the pre-registered fail-fast (fixes <= 2 of P's 84 and agreement <= 0.506)
( for m in 20 40; do
    while [ ! -s "$T/X/snap-$(printf %04d $m).pt" ]; do kill -0 $TX 2>/dev/null || exit 0; sleep 30; done
    sleep 20  # the snapshot is written in one torch.save; give it time to close
    if ! wg $m; then
      [ $m = 40 ] && { alert "it40 window gate failed: stopping training (the fail-fast cannot be checked)"; pkill -f "scripts/train_rl.py"; touch "$T/FAILFAST"; }
      continue
    fi
    if [ $m = 40 ]; then
      python3 - "$T/wg-0040.json" <<'EOF' || { alert "it40 fail-fast: X does not move the windows; stopping training"; pkill -f "scripts/train_rl.py"; touch "$T/FAILFAST"; }
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
sys.exit(1 if s["cand_fixes_of_those"] <= 2 and s["cand_agree"] <= 0.506 else 0)
EOF
    fi
  done ) &
MS=$!
wait $TX; RX=$?
progress "T X exit $RX $(date +%T)"
stop_ext "$T/ext"
kill $MON 2>/dev/null
wait $MS 2>/dev/null
[ -f "$T/FAILFAST" ] && { progress "stopped at it40 by the pre-registered fail-fast"; exit 5; }
[ "$RX" = 0 ] || [ "$RX" = 124 ] || { alert "T X failed ($RX): $(grep -h 'ALERT\|Error' "$T/X.log" | tail -3 | tr '\n' ' ')"; exit 4; }
CK=""; if [ ! -s "$T/X/last.pt" ]; then
  CK=$(ls "$T"/X/snap-*.pt 2>/dev/null | sort | tail -1); [ -n "$CK" ] || { alert "no last.pt and no snapshot"; exit 4; }
  alert "no last.pt (time cap?): exporting $CK"; CK="--ckpt $CK"
fi
# shellcheck disable=SC2086
uv run --no-sync python scripts/tsoj/export_rl.py "$T/X" "$T/export-X" --init $INIT --value-init $VINIT $CK > "$T/export-X.log" 2>&1 \
  || { alert "export X failed: $(tail -2 "$T/export-X.log" | tr '\n' ' ')"; exit 4; }
# final window gate (the primary metric), on the spare cores while the eval runs
OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 uv run --no-sync python -u scripts/audit/window_gate.py --starts data/cache/line_starts/starts-v2.eval.jsonl \
  --cand "$T/export-X/policy/best.pt" --reuse $W/wg/s[0-9]*.json --out "$T/wg-X.json" > "$T/wg-X.txt" 2>&1 &
WGX=$!

# ---- E: held-out tsoj gauntlet for X, paired with the stored A and P ----
t0=$(date +%s); f0=$(date +%Y-%m-%dT%H:%M:%S)
DL_AGENT=raw DL_CKPT="$T/export-X/policy/best.pt" OUTNAME="eval-X" timeout "${CAP_E:-70m}" bash scripts/tsoj/run_arms.sh "$NAME" p 500 \
  "$(wc -l < $B/heldout.jsonl)" "$(( J - 2 ))" $B/heldout.jsonl > "$OUT/run-eval-X.txt" 2>&1 || alert "eval X exit $?"
f1=$(date +%Y-%m-%dT%H:%M:%S)
uv run --no-sync python scripts/tsoj/gauntlet_summary.py "X=$OUT/eval-X.jsonl:$(( $(date +%s) - t0 ))" "A=$W/eval-A.jsonl" "P=$W/eval-P.jsonl" \
  --paired A --out "$OUT/eval.json" > "$OUT/eval.txt" 2>&1 || alert "eval summary failed: $(tail -2 "$OUT/eval.txt" | tr '\n' ' ')"
progress "E: $(grep -h '"X"\|paired' "$OUT/eval.txt" | cut -c1-900 | tr '\n' ' ')"
python3 scripts/tsoj/turn2_interrupts.py "X=$OUT/logs@$f0,$f1" --out "$OUT/turn2.json" > "$OUT/turn2.txt" 2>&1 || alert "turn2 audit failed: $(tail -2 "$OUT/turn2.txt" | tr '\n' ' ')"
progress "turn2: $(cut -c1-600 "$OUT/turn2.txt")"
wait $WGX; r=$?
[ "$r" = 0 ] && progress "WG X: $(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['summary']))" "$T/wg-X.json")" \
  || alert "window gate X exit $r: $(tail -2 "$T/wg-X.txt" | tr '\n' ' ')"
exit 0
