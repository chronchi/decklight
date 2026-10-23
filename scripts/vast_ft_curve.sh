#!/usr/bin/env bash
# Fine-tune learning curve: fine-tune one deck (scripts/finetune_deck.py) saving snapshots,
# play a validation card (deal seeds 7000+) on every snapshot, pick the best snapshot
# (scripts/audit/ft_select.py), confirm it on a test card (deal seeds 9000+).
# Pre-registered: NOTEBOOK 2026-10-22 fine-tune learning curve.
# Env: DECK (BlueEyes), FT_HOURS (2), SNAP_EVERY (40), SHARE (0.5), VAL_SEED0 (7000),
#   VAL_SEEDS (6), TEST_SEED0 (9000), TEST_SEEDS (8), SMOKE (0; SMOKE: --iters 4,
#   SNAP_EVERY 2, 1 seed per card), NP (empty = auto), GB_PER_PROC (0.6, GPU GB per card
#   process), RAM_PER_PROC (3, GB), MS_IT (20), MS_MIN (0.45), DRY (1 = echo every command,
#   run nothing).
# Usage: bash scripts/vast_ft_curve.sh <name>. Do NOT run on the laptop (vast GPU only).
#
# Seed ranges (deal seeds): deck_card gives gauntlet opponent k (of the non-target decks,
# in sorted-stem order) the seeds seed0+k*seeds .. seed0+(k+1)*seeds-1, i.e. the card's
# seed interval is [seed0, seed0+NOPP*seeds). With the 31-deck gauntlet (NOPP=30):
# validation [7000, 7000+30*6) = 7000..7179, test [9000, 9000+30*8) = 9000..9239 --
# disjoint by construction (the test interval starts 1820 above validation's end).
# SMOKE (1 seed/card): 7000..7029 vs 9000..9029, still disjoint. The script recomputes
# both intervals from the env + the actual gauntlet size and exits non-zero (alert)
# if they overlap.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_ft_curve.sh <name>}"
OUT="runs/$NAME"
mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
tick() { echo "$(date +%T) $1" | tee -a "$OUT/PROGRESS"; }
DECK=${DECK:-BlueEyes}
FT_HOURS=${FT_HOURS:-2}; SNAP_EVERY=${SNAP_EVERY:-40}; SHARE=${SHARE:-0.5}
VAL_SEED0=${VAL_SEED0:-7000}; VAL_SEEDS=${VAL_SEEDS:-6}
TEST_SEED0=${TEST_SEED0:-9000}; TEST_SEEDS=${TEST_SEEDS:-8}
SMOKE=${SMOKE:-0}
NP=${NP:-}; GB_PER_PROC=${GB_PER_PROC:-0.6}; RAM_PER_PROC=${RAM_PER_PROC:-3}
MS_IT=${MS_IT:-20}; MS_MIN=${MS_MIN:-0.45}
DRY=${DRY:-0}
if [ "$SMOKE" = 1 ]; then SNAP_EVERY=2; VAL_SEEDS=1; TEST_SEEDS=1; fi
FAILED=0

# DRY=1: echo every `uv run` command exactly, skip setup/watcher/sizing (NP=4).
runv() {
  if [ "$DRY" = 1 ]; then
    echo "+ uv run --no-sync python $*"
  else
    uv run --no-sync python "$@"
  fi
}

if [ "$DRY" != 1 ]; then
  # Setup exactly like vast_stage_p.sh.
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
  USAGE_PID=$!

  OPPD=$(( $(ls decks/gauntlet/*.ydk | wc -l) - 1 ))
  # NP sizing: NP = max(1, min(J, floor(gpu*0.85/GB_PER_PROC), floor(ram*0.8/RAM_PER_PROC))),
  # capped at the smaller card's deal count so no proc gets 0 deals.
  if [ -z "$NP" ]; then
    GTOT_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    RTOT_GB=$(free -g | awk '/Mem/{print $2}')
    VAL_DEALS=$((OPPD * 2 * VAL_SEEDS)); TEST_DEALS=$((OPPD * 2 * TEST_SEEDS))
    if [ "$TEST_DEALS" -lt "$VAL_DEALS" ]; then CAP_DEALS=$TEST_DEALS; else CAP_DEALS=$VAL_DEALS; fi
    NP=$(python3 -c "print(max(1, min($J, int($GTOT_MB / 1024 * 0.85 / $GB_PER_PROC), int($RTOT_GB * 0.8 / $RAM_PER_PROC), $CAP_DEALS)))")
    {
      echo "cores $J, gpu ${GTOT_MB}MB, ram ${RTOT_GB}GB; per-proc ${GB_PER_PROC}GB/${RAM_PER_PROC}GB -> NP=$NP (deal cap $CAP_DEALS)"
    } | tee "$OUT/np.txt"
  else
    echo "NP=$NP (env)" | tee "$OUT/np.txt"
  fi
else
  NP=${NP:-4}
  OPPD=$(( $(ls decks/gauntlet/*.ydk 2>/dev/null | wc -l) - 1 ))
  [ "$OPPD" -ge 1 ] 2>/dev/null || OPPD=30
fi

watch_ft() { # <ftdir> <tag>: tail metrics.jsonl -> PROGRESS every 30 s + the MS_IT milestone (alert only).
  local dir="$1" tag="$2" seen=0 ms_done=0 m="$1/rl/metrics.jsonl" out maxit
  while true; do
    sleep 30
    [ -f "$m" ] || continue
    out=$(uv run --no-sync python scripts/audit/ft_progress.py "$m" --tag "$tag" \
      --seen "$seen" --ms-it "$MS_IT" --ms-min "$MS_MIN" 2>&1)
    rc=$?
    echo "$out" | grep -h "^PROGRESS" >> "$OUT/PROGRESS" 2>/dev/null || true
    echo "$out" | grep -h "^MILESTONE " >> "$OUT/PROGRESS" 2>/dev/null || true
    maxit=$(echo "$out" | grep -oh "it=[0-9]*" | grep -oh "[0-9]*" | sort -n | tail -1)
    [ -n "${maxit:-}" ] && seen=$maxit
    if [ $rc = 2 ] && [ $ms_done = 0 ]; then
      ms_done=1
      echo "$out" | grep -h "^MILESTONE_FAIL" >> "$OUT/ALERT" 2>/dev/null || true
      alert "$tag milestone it$MS_IT mean focus_wr_vs_bc below $MS_MIN (run continues)"
    fi
    [ -f "$dir/DONE_FT" ] && break
  done
}

# Validation/test deal seeds must not overlap (deck_card: opp k gets seed0+k*seeds..).
VAL_LO=$VAL_SEED0; VAL_HI=$((VAL_SEED0 + OPPD * VAL_SEEDS - 1))
TEST_LO=$TEST_SEED0; TEST_HI=$((TEST_SEED0 + OPPD * TEST_SEEDS - 1))
echo "seeds: val $VAL_LO..$VAL_HI, test $TEST_LO..$TEST_HI ($OPPD opps)" | tee -a "$OUT/PROGRESS"
if [ "$VAL_LO" -le "$TEST_HI" ] && [ "$TEST_LO" -le "$VAL_HI" ]; then
  alert "validation seeds $VAL_LO..$VAL_HI overlap test seeds $TEST_LO..$TEST_HI"
  exit 3
fi

card_progress_all() { # <tag> <dir>: one "PROGRESS <tag> <arm> diff=<d> ci=[lo,hi] n=.." per non-base arm.
  local tag="$1" dir="$2"
  uv run --no-sync python - "$tag" "$dir/card.json" >> "$OUT/PROGRESS" 2>"$dir/progress.txt" <<'EOF' || return 1
import json, sys
from pathlib import Path
tag, path = sys.argv[1], sys.argv[2]
card = json.loads(Path(path).read_text())
first = card["diff_vs_first"]["first"]
for name, d in card["diff_vs_first"].items():
    if name == "first":
        continue
    print(f"PROGRESS {tag} {name} diff={d['diff_pp']} ci=[{d['ci_pp'][0]},{d['ci_pp'][1]}] n={d['n']}")
EOF
}

run_card() { # <tag> <seed0> <seeds>: plays the global ARMSPEC arms (name=ckpt:value), then summarizes.
  local tag="$1" seed0="$2" seeds="$3"
  local dir="$OUT/$tag"
  mkdir -p "$dir"
  local deck="decks/gauntlet/$DECK.ydk"
  local flags=() a
  for a in "${ARMSPEC[@]}"; do flags+=(--arm "$a"); done
  local p pid rc procs_failed=0
  if [ "$DRY" = 1 ]; then
    for ((p = 0; p < NP; p++)); do
      runv scripts/deck_card.py --deck "$deck" "${flags[@]}" \
        --search 0 --deal-seed0 "$seed0" --seeds "$seeds" --device cuda --out "$dir" \
        --nproc "$NP" --proc "$p"
    done
    runv scripts/deck_card.py --deck "$deck" "${flags[@]}" \
      --search 0 --deal-seed0 "$seed0" --seeds "$seeds" --device cuda --out "$dir" \
      --nproc "$NP" --proc 0 --summarize
    return 0
  fi
  local pids=()
  for ((p = 0; p < NP; p++)); do
    # one torch thread per card proc (NP procs share the cores); the trainer keeps its defaults
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 runv scripts/deck_card.py --deck "$deck" "${flags[@]}" \
      --search 0 --deal-seed0 "$seed0" --seeds "$seeds" --device cuda --out "$dir" \
      --nproc "$NP" --proc "$p" > "$dir/proc-$p.txt" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid"; rc=$?
    if [ "$rc" != 0 ]; then
      alert "$tag: proc pid $pid exit $rc"
      procs_failed=1
    fi
  done
  grep -h "^ALERT\|Traceback" "$dir"/proc-*.txt >> "$OUT/ALERT" 2>/dev/null || true
  if [ $procs_failed = 1 ]; then
    return 1
  fi
  # --summarize reuses the finished shards (same --nproc/--proc 0 passes the reuse check), plays nothing new.
  if ! runv scripts/deck_card.py --deck "$deck" "${flags[@]}" \
      --search 0 --deal-seed0 "$seed0" --seeds "$seeds" --device cuda --out "$dir" \
      --nproc "$NP" --proc 0 --summarize > "$dir/summarize.txt" 2>&1; then
    alert "$tag: summarize failed: $(tail -1 "$dir/summarize.txt")"
    FAILED=1
    return 0
  fi
  if ! card_progress_all "$tag" "$dir"; then
    alert "$tag: PROGRESS lines failed: $(tail -1 "$dir/progress.txt")"
    FAILED=1
  fi
  return 0
}

snap_arms() { # prints "name=policy:value" per snapshot arm (base first, final last), sha256-deduped.
  python3 - "$OUT/ft-$DECK/FINETUNE.json" "$OUT/ft-$DECK" <<'EOF'
import hashlib, json, sys
fin_path, ftdir = sys.argv[1], sys.argv[2]
fin = json.loads(open(fin_path).read())
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
cands = [("base", "data/cache/P/best.pt", "data/cache/value_p2/best.pt")]
for s in fin.get("snapshots", []):
    it = s["iter"]
    cands.append((f"it{it:04d}", f"{ftdir}/snaps/it-{it:04d}/policy/best.pt",
                  f"{ftdir}/snaps/it-{it:04d}/value/best.pt"))
cands.append(("final", f"{ftdir}/policy/best.pt", f"{ftdir}/value/best.pt"))
seen = {}
for name, p, v in cands:
    h = sha(p)
    if h in seen:
        print(f"# drop {name}: same policy sha256 as {seen[h]}", file=sys.stderr)
        continue
    seen[h] = name
    print(f"{name}={p}:{v}")
EOF
}

# ---- 1. fine-tune with snapshots ----
FT_EXTRA=()
if [ "$SMOKE" = 1 ]; then FT_EXTRA=(--iters 4); fi
if [ "$DRY" = 1 ]; then
  runv scripts/finetune_deck.py --deck "decks/gauntlet/$DECK.ydk" --out "$OUT/ft-$DECK" \
    --hours "$FT_HOURS" --snap-every "$SNAP_EVERY" --share "$SHARE" --device cuda "${FT_EXTRA[@]}"
else
  if [ ! -f "decks/gauntlet/$DECK.ydk" ]; then
    alert "missing decks/gauntlet/$DECK.ydk"
    exit 3
  fi
  rm -f "$OUT/ft-$DECK/DONE_FT"
  watch_ft "$OUT/ft-$DECK" "ft-$DECK" &
  WPID=$!
  if runv scripts/finetune_deck.py --deck "decks/gauntlet/$DECK.ydk" --out "$OUT/ft-$DECK" \
      --hours "$FT_HOURS" --snap-every "$SNAP_EVERY" --share "$SHARE" --device cuda \
      "${FT_EXTRA[@]}" > "$OUT/ft-$DECK.log" 2>&1; then
    tick "ft-$DECK done"
  else
    alert "ft-$DECK failed: $(tail -1 "$OUT/ft-$DECK.log")"
    touch "$OUT/ft-$DECK/DONE_FT"
    kill "$WPID" 2>/dev/null || true
    wait "$WPID" 2>/dev/null || true
    exit 3  # a failed stage skips the stages that depend on it: nothing after runs
  fi
  touch "$OUT/ft-$DECK/DONE_FT"
  kill "$WPID" 2>/dev/null || true
  wait "$WPID" 2>/dev/null || true
  # train_rl's optimizer state only: drop last.pt when it is > 200 MB; never touch policy/, value/ or snaps/.
  if [ -f "$OUT/ft-$DECK/rl/last.pt" ] && [ "$(stat -c%s "$OUT/ft-$DECK/rl/last.pt")" -gt 209715200 ]; then
    rm -f "$OUT/ft-$DECK/rl/last.pt"
  fi
fi

# ---- 2. validation card: base + every snapshot + final (deduped) ----
if [ "$DRY" = 1 ]; then
  # FINETUNE.json does not exist in DRY: assume 4 snapshots it0040..it0160 plus the final.
  ARMSPEC=(base=data/cache/P/best.pt:data/cache/value_p2/best.pt
    it0040="$OUT/ft-$DECK/snaps/it-0040/policy/best.pt:$OUT/ft-$DECK/snaps/it-0040/value/best.pt"
    it0080="$OUT/ft-$DECK/snaps/it-0080/policy/best.pt:$OUT/ft-$DECK/snaps/it-0080/value/best.pt"
    it0120="$OUT/ft-$DECK/snaps/it-0120/policy/best.pt:$OUT/ft-$DECK/snaps/it-0120/value/best.pt"
    it0160="$OUT/ft-$DECK/snaps/it-0160/policy/best.pt:$OUT/ft-$DECK/snaps/it-0160/value/best.pt"
    final="$OUT/ft-$DECK/policy/best.pt:$OUT/ft-$DECK/value/best.pt")
else
  mapfile -t ARMSPEC < <(snap_arms 2> "$OUT/dedup.txt" | grep -v "^#")
  cat "$OUT/dedup.txt" >> "$OUT/PROGRESS" 2>/dev/null || true
  if [ "${#ARMSPEC[@]}" -lt 2 ]; then
    alert "card-val: no snapshot/final arms (only base); skipping the rest"
    exit 3
  fi
fi
if ! run_card "card-val" "$VAL_SEED0" "$VAL_SEEDS"; then
  alert "card-val failed; select + test card skipped"
  exit 3
fi
if [ "$DRY" != 1 ] && [ ! -f "$OUT/card-val/card.json" ]; then
  alert "card-val has no card.json; select + test card skipped"
  exit 3
fi

# ---- 3. select the best snapshot on validation ----
if [ "$DRY" = 1 ]; then
  runv scripts/audit/ft_select.py --card "$OUT/card-val/card.json" --base base --out "$OUT/select.json"
  BEST=it0080  # DRY assumption for the test-card trace below
else
  if ! runv scripts/audit/ft_select.py --card "$OUT/card-val/card.json" --base base \
      --out "$OUT/select.json" > "$OUT/select.txt" 2>&1; then
    alert "select failed: $(tail -1 "$OUT/select.txt")"
    exit 3  # the test card needs the selected arm
  fi
  grep -h "^PROGRESS" "$OUT/select.txt" >> "$OUT/PROGRESS" 2>/dev/null || true
  BEST=$(python3 -c "import json; print(json.load(open('$OUT/select.json'))['best'])")
fi

# ---- 4. test card: base + best (+ final when best != final) ----
if [ "$DRY" = 1 ]; then
  ARMSPEC=(base=data/cache/P/best.pt:data/cache/value_p2/best.pt
    "$BEST=$OUT/ft-$DECK/snaps/it-0080/policy/best.pt:$OUT/ft-$DECK/snaps/it-0080/value/best.pt"
    final="$OUT/ft-$DECK/policy/best.pt:$OUT/ft-$DECK/value/best.pt")
else
  ARMSPEC=()
  while IFS= read -r spec; do
    name="${spec%%=*}"
    if [ "$name" = "base" ] || [ "$name" = "$BEST" ] || { [ "$name" = "final" ] && [ "$BEST" != "final" ]; }; then
      ARMSPEC+=("$spec")
    fi
  done < <(snap_arms 2>/dev/null | grep -v "^#")
fi
if ! run_card "card-test" "$TEST_SEED0" "$TEST_SEEDS"; then
  alert "card-test failed"
  FAILED=1
fi
if [ "$DRY" != 1 ] && [ ! -f "$OUT/card-test/card.json" ]; then
  alert "card-test has no card.json"
  FAILED=1
fi

if [ "$DRY" = 1 ]; then
  echo '{"dry": true}' > "$OUT/curve.json"
  cat "$OUT/curve.json"
  exit 0
fi

# ---- 5. collect everything into curve.json; exit 0 only when every stage ran ----
if [ ! -f "$OUT/select.json" ] || [ ! -f "$OUT/card-test/card.json" ]; then
  FAILED=1
fi
if ! uv run --no-sync python - "$OUT" "ft-$DECK" <<'EOF'; then
import json, sys
from pathlib import Path
out, ftdir = Path(sys.argv[1]), sys.argv[2]
curve: dict = {}
curve["finetune"] = json.loads((out / ftdir / "FINETUNE.json").read_text())
for k in ("card-val", "card-test"):
    curve[k] = json.loads((out / k / "card.json").read_text())
curve["select"] = json.loads((out / "select.json").read_text())
(out / "curve.json").write_text(json.dumps(curve, indent=1))
print(json.dumps(curve["select"]))
EOF
  alert "curve.json failed"
  FAILED=1
fi
if [ "$DRY" != 1 ]; then
  kill "$USAGE_PID" 2>/dev/null || true
fi
[ $FAILED = 0 ]
