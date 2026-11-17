#!/usr/bin/env bash
# tsoj PPO, steps 2+3 (NOTEBOOK 2026-11-07 tsoj PPO pre-registration): PPO from P (forge_r1) on one GPU box.
#   arm A: league (self-play, frozen P, the 16 specialists piloting their own decks) + tsoj's bot (500 nodes) as
#          EXT_SHARE of every batch, through the arbiter: actors = remote clients -> policy server (GPU) -> episode files
#          -> train_rl --ext-dir;
#   arm B: the same league without tsoj's bot.
# Stages: S (smoke: 3 iterations of each arm with every invariant checked, else stop), T (A then B, ITERS each;
# one learner at a time: two do not fit a 24 GB GPU), E (held-out tsoj gauntlet, seeds 20000+, mirror book, 500 nodes, raw argmax: P, A, B on the same openings).
# Env: ITERS (80), EXT_SHARE (0.2), TRAIN_SEEDS (300), EVAL_SEEDS (28), CAP_T (5h, arm A), CAP_TB (2h, arm B; B runs as many iterations as A finished), CAP_E (70m per arm), LRES (10
# cores kept for the learners + server). Usage: bash scripts/vast_tsoj_ppo.sh <name>
set -uo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
# OUT is absolute: the client (scripts/tsoj/client/WindBot) runs in its own directory and refuses relative DL_ paths
NAME="$1"; OUT="$PWD/runs/tsoj-arms/$NAME"; mkdir -p "$OUT" "runs/$NAME"
# the launcher pulls runs/$NAME: copy everything except raw client streams (records/*.stream.jsonl.gz) on any exit
trap 'rsync -a --exclude "*.stream.jsonl.gz" "$OUT"/ "runs/$NAME"/ 2>/dev/null || cp -r "$OUT"/. "runs/$NAME"/; pkill -P $$ 2>/dev/null' EXIT
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
progress() { echo "$*" | tee -a "$OUT/PROGRESS"; }
export OMP_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
# 51 bots loading at once next to two learners need > 30 s to answer isready (tsoj-ppo-20261107-1718)
READY_TIMEOUT=300 TASKSET="" CARGO_BUILD_JOBS="$J" bash scripts/tsoj/build_tsoj.sh > "$OUT/build.log" 2>&1 || { alert "build_tsoj failed: $(tail -3 "$OUT/build.log" | tr '\n' ' ')"; exit 3; }
progress "build done $(date +%T)"

B=scripts/tsoj/books
uv run --no-sync python scripts/tsoj/make_book.py --mirror --seeds "${TRAIN_SEEDS:-300}" --seed-base 100000 --out $B/train.jsonl | head -1 | tee -a "$OUT/PROGRESS"
uv run --no-sync python scripts/tsoj/make_book.py --mirror --seeds "${EVAL_SEEDS:-28}" --seed-base 20000 --out $B/heldout.jsonl | head -1 | tee -a "$OUT/PROGRESS"
# a remote game keeps ~0.5 core busy (bot and P alternate; P's forward is on the GPU): 20 games on 30 spare cores
# left load at 10-17 (tsoj-ppo-20261107-1748), so ACT_PER_CORE (2) games per spare core
NA=$(( (J - ${LRES:-10}) * ${ACT_PER_CORE:-2} )); [ "$NA" -lt 8 ] && { alert "only $J cores: $NA actor slots"; exit 3; }
echo "cores $J -> $NA actor games at once, ${LRES:-10} cores for learners + server" | tee "$OUT/sizing.txt"
MANI=data/cache/deck_opps/manifest.json
[ -s "$MANI" ] || { alert "missing $MANI"; exit 3; }
SOCK=/tmp/dlp-$$.sock

start_ext() {  # start_ext DIR: policy server on DIR/weights.pt + actor loop writing episodes to DIR
  local d="$1"; mkdir -p "$d"; rm -f "$d/ready" "$d/STOP"
  PYTHONUNBUFFERED=1 uv run --no-sync python -u scripts/tsoj/policy_server.py --sock "$SOCK" --weights "$d/weights.pt" \
    --value-init data/cache/value_p2/best.pt --device cuda --ready "$d/ready" > "$d/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 120); do [ -f "$d/ready" ] && break; sleep 1; done
  [ -f "$d/ready" ] || { alert "policy server not ready: $(tail -3 "$d/server.log" | tr '\n' ' ')"; return 1; }
  # watchdog: a chunk with games but no new episodes, or 2 chunks in a row with no games, kills the learners
  # (their non-zero exit stops the job) instead of letting train_rl wait ext_wait_s x ext_zero_iters
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
stop_ext() {  # stop_ext DIR
  touch "$1/STOP"; pkill -f "ygo-arbiter" 2>/dev/null; sleep 5
  kill "$ACTORS_PID" "$SERVER_PID" 2>/dev/null; wait "$ACTORS_PID" "$SERVER_PID" 2>/dev/null
  grep -h "^ALERT" "$1/actors.log" "$1/server.log" 2>/dev/null | sort | uniq -c | sed 's/^/ALERT ext: /' >> "$OUT/ALERT"
}
COMMON="--init data/cache/forge_r1/best.pt --value-init data/cache/value_p2/best.pt --lr 4e-5 --kl-coef 0.1
  --eval-every 1000000000 --device cuda --no-ygoagent --league 1 --p-self 0.4 --p-ygoagent 0 --deck-opps $MANI
  --threads 4 --hours 8"
train() {  # train OUT ITERS [ext args...]; TCAP = time cap (0 = none)
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

# ---- S: smoke (first milestone) ----
S="$OUT/smoke"; mkdir -p "$S"
start_ext "$S/ext" || exit 4
# the arms run one after the other: one learner holds ~10 GB of GPU memory and two OOM'd a 24 GB card at
# iteration 3 of T (tsoj-ppo-20261107-1847)
train "$S/A" 3 --ext-dir "$S/ext" --ext-share "${EXT_SHARE:-0.2}" --ext-wait-s 900; RA=$?
stop_ext "$S/ext"
train "$S/B" 3; RB=$?
progress "smoke ext chunks: $(tr '\n' ';' < "$S/ext/chunks.txt" 2>/dev/null | cut -c1-500)"
progress "smoke A exit $RA: $(metric "$S/A/metrics.jsonl" 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-600)"
progress "smoke B exit $RB: $(metric "$S/B/metrics.jsonl" 2>&1 | tail -1 | cut -c1-400)"
grep -h "deck-opps\|\[ext\]" "$S/A.log" "$S/B.log" | head -3 | tee -a "$OUT/PROGRESS"
ok=1
[ "$RA" = 0 ] && [ "$RB" = 0 ] || { alert "smoke train_rl exit A=$RA B=$RB: $(grep -h 'ALERT\|Error' "$S/A.log" "$S/B.log" | tail -3 | tr '\n' ' ')"; ok=0; }
python3 - "$S/A/metrics.jsonl" "$S/B/metrics.jsonl" "$S/A.log" "$S/B.log" <<'EOF' || ok=0
import json, re, sys
# training records only: iteration 0 and the eval rows ({"iter", "vs_bc", ...}) have no rollout ("samples")
def train_rows(path):
    return [r for r in (json.loads(x) for x in open(path) if '"iter"' in x) if r["iter"] > 0 and "samples" in r]
rec = train_rows(sys.argv[1])
bad = []
if len(rec) < 3: bad.append(f"A wrote {len(rec)} iterations")
if any(r.get("ext_games", 0) == 0 for r in rec): bad.append("an A iteration had 0 tsoj games")
if any(r.get("ext_stale", 0) > r.get("ext_games", 0) for r in rec): bad.append("more stale than used tsoj games")
for m in sys.argv[1:3]:
    it = train_rows(m)
    if len(it) < 3 or any(r.get("deck_opp_games", 0) == 0 for r in it): bad.append(f"{m}: an iteration had 0 deck-bound specialist games")
    print(m, "deck_opp games/wr:", [(r.get("deck_opp_games"), r.get("deck_opp_wr")) for r in it])
for log in sys.argv[3:]:
    t = open(log).read()
    if not re.search(r"\[deck-opps\] 16 nets, 48 bound decks", t): bad.append(f"{log}: deck-opps not loaded as 16/48")
for b in bad: print("ALERT smoke:", b)
sys.exit(1 if bad else 0)
EOF
[ "$ok" = 1 ] || { alert "smoke failed: stop before training"; grep -h "^ALERT" "$OUT/PROGRESS" >> "$OUT/ALERT"; exit 4; }
progress "smoke OK $(date +%T)"

eval_arm() {  # eval_arm P|A|B CONC: held-out tsoj gauntlet with the raw policy (P = forge_r1, the client default)
  local arm="$1" ck="" t0; [ "$arm" != P ] && ck="$T/export-$arm/policy/best.pt"
  [ -n "$ck" ] && [ ! -s "$ck" ] && { alert "no checkpoint for $arm"; return 1; }
  t0=$(date +%s)
  DL_AGENT=raw DL_CKPT="$ck" OUTNAME="eval-$arm" timeout "${CAP_E:-70m}" bash scripts/tsoj/run_arms.sh "$NAME" p 500 \
    "$(wc -l < $B/heldout.jsonl)" "$2" $B/heldout.jsonl > "$OUT/run-eval-$arm.txt" 2>&1 || alert "eval $arm exit $?"
  echo "$arm=$OUT/eval-$arm.jsonl:$(( $(date +%s) - t0 ))" >> "$OUT/specs-eval.txt"
  progress "E $arm: $(uv run --no-sync python scripts/tsoj/gauntlet_summary.py "500=$OUT/eval-$arm.jsonl" 2>&1 | tail -1 | cut -c1-300)"
}

# ---- T: train A and B ----
T="$OUT/train"; mkdir -p "$T"
start_ext "$T/ext" || exit 4
( while sleep 300; do
    for arm in A B; do [ -f "$T/$arm/metrics.jsonl" ] && progress "T $arm $(metric "$T/$arm/metrics.jsonl" | tail -1 | cut -c1-500)"; done
  done ) &
MON=$!
TCAP="${CAP_T:-5h}" train "$T/A" "${ITERS:-80}" --ext-dir "$T/ext" --ext-share "${EXT_SHARE:-0.2}"; RA=$?
progress "T A exit $RA $(date +%T)"
stop_ext "$T/ext"
[ "$RA" = 0 ] || [ "$RA" = 124 ] || { kill $MON 2>/dev/null; alert "T A failed ($RA): $(grep -h 'ALERT\|Error' "$T/A.log" | tail -3 | tr '\n' ' ')"; exit 4; }
# P's held-out eval needs neither the GPU nor the arms: run it on the spare cores while B trains
eval_arm P "$(( J - ${LRES:-10} ))" &
PE=$!
# B trains exactly as many iterations as A finished (A may stop at CAP_T): A - B then compares equal update counts
NB=$(python3 -c "import json,sys; print(sum(1 for x in open(sys.argv[1]) if '\"samples\"' in x and json.loads(x).get('iter', 0) > 0))" "$T/A/metrics.jsonl")
[ "${NB:-0}" -gt 0 ] || { alert "A finished 0 training iterations"; exit 4; }
progress "T B: $NB iterations (= A's) $(date +%T)"
TCAP="${CAP_TB:-2h}" train "$T/B" "$NB"; RB=$?
progress "T B exit $RB $(date +%T)"
kill $MON 2>/dev/null
wait $PE
# 124 = the CAP_T time cap (last.pt is still a valid arm); anything else is a failed arm: do not pay for the eval
for r in "$RA" "$RB"; do
  [ "$r" = 0 ] || [ "$r" = 124 ] || { alert "T failed (A=$RA B=$RB): $(grep -h 'ALERT\|Error' "$T/A.log" "$T/B.log" | tail -3 | tr '\n' ' ')"; exit 4; }
done
for arm in A B; do
  uv run --no-sync python scripts/tsoj/export_rl.py "$T/$arm" "$T/export-$arm" > "$T/export-$arm.log" 2>&1 \
    || alert "export $arm failed: $(tail -2 "$T/export-$arm.log" | tr '\n' ' ')"
done

# ---- E: held-out tsoj gauntlet (same openings for P, A, B; P's runs during T/B) ----
for arm in A B; do eval_arm "$arm" "$J"; done
uv run --no-sync python scripts/tsoj/gauntlet_summary.py $(cat "$OUT/specs-eval.txt") --paired P --out "$OUT/eval.json" > "$OUT/eval.txt" 2>&1 \
  || alert "eval summary failed: $(tail -2 "$OUT/eval.txt" | tr '\n' ' ')"
progress "E paired: $(grep -h paired "$OUT/eval.txt" | cut -c1-600)"
exit 0
