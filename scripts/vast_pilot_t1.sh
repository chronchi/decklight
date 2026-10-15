#!/usr/bin/env bash
# Plan-v4 pilot, tranche 1 (docs/retrain/plan-v4-draft.md §6; user 2026-10-15: "Let's go with the test, just added some
# credits"): can a fresh BC student distilled from P (P's sampled games on all train decks, CE on P's move + KL to P's
# full distribution) play at P's level?
#   1. G: GAMES P games (temp 1.0) with gen_selfplay, our side on train pool decks only; league self (P argmax) 0.5,
#      ygo-agent 0.3, B-v1 0.1, base_r3 0.1; duel seeds 500000+ (one block of 10000 per process).
#      Asserts: every train deck >= 30 student games; no holdout / new-card / gauntlet deck on the student seat;
#      >= 95% of the games written; cut share < 2%.
#   2. shards (kind pgame) and the student: train_bc from scratch, P's architecture, --kl-ref P (coef 1), EPOCHS,
#      --select-on last. First milestone (epoch 1): val loss must fall from epoch 0, else stop.
#   3. raw card on 5,952 fresh deals (deal seeds 7000+): P and the student vs ygo-agent, paired; h2h student vs P on
#      train-pool, holdout and new-card decks; principles (gates 3-4) for the student.
#   Gate (pre-registered): student >= P - 2 pp (paired), each seat >= -3 pp; h2h vs P >= 0.45. Fail -> option B closed.
# Env: GAMES (20000), EPOCHS (4), GA_SEEDS (24), VAR_CAP (60m), SMOKE=1 (tiny everything).
# Usage: bash scripts/vast_pilot_t1.sh <name>. vast GPU only (~40 GB RAM for training).
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }
P=data/cache/P/best.pt
for f in "$P" data/cache/P/metrics.jsonl data/cache/bv1/best.pt data/cache/base_r3/best.pt; do
  [ -s "$f" ] || { alert "missing $f"; exit 3; }
done
GAMES=${GAMES:-20000}; EPOCHS=${EPOCHS:-4}; GS=${GA_SEEDS:-24}; LIM=()
[ "${SMOKE:-0}" = 1 ] && { GAMES=$((J * 4)); EPOCHS=2; GS=1; LIM=(--limit 20000); }

# 1. G: P's sampled games, one process per core (each its own seed block)
# processes: min(cores, GPU memory / ~1.5 GB (CUDA context + 4 small policies), RAM / ~3 GB); measured in gen-0.txt usage
GMB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); RMB=$(free -m | awk '/Mem/{print $2}')
NP=$(python3 -c "print(max(1, min($J, $GMB // 1500, $RMB // 3000)))"); echo "G: $NP procs (cores $J, gpu ${GMB}MB, ram ${RMB}MB)" | tee -a "$OUT/PROGRESS"
PER=$(( (GAMES + NP - 1) / NP )); [ "$PER" -le 10000 ] || { alert "$PER games per process > 10000 (seed blocks overlap)"; exit 3; }
LEAGUE="self:0.5,ygoagent:0.3,bv1=data/cache/bv1/best.pt:0.1,base=data/cache/base_r3/best.pt:0.1"
mkdir -p "$OUT/pgames"
PIDS=()
for ((p = 0; p < NP; p++)); do
  uv run --no-sync python scripts/gen_selfplay.py --ckpt "$P" --out "$OUT/pgames/p$p" --games "$PER" --temp 1.0 \
    --league "$LEAGUE" --seed $((500000 + 10000 * p)) --threads 1 --device cuda > "$OUT/pgames/gen-$p.txt" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || alert "gen proc $pid exit $?"; done
tick "G $(cat "$OUT"/pgames/p*/*.jsonl 2>/dev/null | wc -l) games"
uv run --no-sync python - "$OUT" "$GAMES" "${SMOKE:-0}" <<'PY' >> "$OUT/PROGRESS" || { alert "G asserts failed: $(tail -1 "$OUT/PROGRESS")"; exit 4; }
import collections, glob, json, sys
from pathlib import Path
out, want, smoke = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
hold = set(Path("decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
train = {p.stem for p in Path("decks/pool").glob("*.ydk")} - hold
banned = hold | {p.stem for p in Path("decks/newcards").glob("*.ydk")} | {p.stem for p in Path("decks/gauntlet").glob("*.ydk")}
n = cut = 0
per_deck, opp_g, opp_w = collections.Counter(), collections.Counter(), collections.Counter()
for f in glob.glob(str(out / "pgames/p*/*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        n += 1
        stu = r["decks"][r["student_seat"]]
        if stu in banned - train:
            sys.exit(f"ALERT student deck {stu} is holdout / new-card / gauntlet")
        per_deck[stu] += 1
        opp_g[r["opponent"]] += 1
        cut += r["winner"] == 3
        opp_w[r["opponent"]] += r["winner"] == r["student_seat"]
low = sorted(d for d in train if per_deck[d] < (1 if smoke else 30))
wr = " ".join(f"{o} {opp_w[o] / opp_g[o]:.3f} ({opp_g[o]})" for o in sorted(opp_g))
print(f"G: {n} games, {len(per_deck)} student decks (min {min(per_deck.values())}), cut {cut}; P sampled win rate: {wr}")
if n < 0.95 * want or (not smoke and (low or cut > 0.02 * n)):
    sys.exit(f"ALERT G: {n}/{want} games, {len(low)} train decks under 30 games {low[:5]}, cut {cut}")
PY

# 2. shards + the student
uv run --no-sync python scripts/build_bc_shards.py --glob "$OUT/pgames/p*/*.jsonl" --out data/cache/bc_shards_p --jobs "$J" \
  > "$OUT/shards.txt" 2>&1 || { alert "shards failed: $(tail -1 "$OUT/shards.txt")"; exit 3; }
grep -h "^ALERT" "$OUT/shards.txt" >> "$OUT/ALERT"
tick "shards $(grep -m1 '^samples' "$OUT/shards.txt")"
HOLD="kewl-tune $(tr '\n' ' ' < decks/pool/HOLDOUT)"
# shellcheck disable=SC2086
uv run --no-sync python scripts/train_bc.py --shards data/cache/bc_shards_p --emb data/cards/emb/qwen3-0.6b-d512.npz \
  --out "$OUT/student" --epochs "$EPOCHS" --batch 512 --lr 5e-4 --holdout $HOLD --select-on last --kinds pgame \
  --require-kinds pgame --kl-ref "$P" --kl-ref-coef 1.0 "${LIM[@]}" > "$OUT/train.txt" 2>&1 &
TP=$!
( seen=0; while kill -0 $TP 2>/dev/null; do  # relay each epoch; first milestone: val loss falls from epoch 0 to 1
    sleep 60; n=$(grep -c '"epoch"' "$OUT/student/metrics.jsonl" 2>/dev/null); n=${n:-0}
    if [ "$n" -gt "$seen" ]; then
      echo "$(date +%T) train $(tail -1 "$OUT/student/metrics.jsonl" | cut -c1-400)" >> "$OUT/PROGRESS"; seen=$n
      if [ "$n" = 2 ] && ! uv run --no-sync python -c "
import json, sys
r = [json.loads(x) for x in open('$OUT/student/metrics.jsonl') if '\"val\"' in x]
sys.exit(0 if r[1]['val']['loss'] < r[0]['val']['loss'] else 1)"; then
        alert "student val loss did not fall from epoch 0 to 1; stopping"; kill $TP
      fi
    fi
  done ) &
wait $TP; rc=$?
rm -f "$OUT/student/last.pt"
grep -h "^ALERT" "$OUT/train.txt" >> "$OUT/ALERT"
[ $rc -eq 0 ] && [ -s "$OUT/student/best.pt" ] || { alert "student training failed (exit $rc): $(tail -1 "$OUT/train.txt")"; exit 4; }
tick "student"
rm -rf data/cache/bc_shards_p

# 3. raw card (deal seeds 7000+), paired; h2h vs P on train-pool sample, holdout and new-card decks
S="$OUT/student/best.pt"
V="praw|--opp ygoagent --raw 1 --ga-seeds $GS"
V="$V;student|--opp ygoagent --raw 1 --ga-seeds $GS --ckpt $S"
V="$V;h2h|--opp ckpt --opp-ckpt $P --raw 1 --decks 16 --seeds 4 --ckpt $S"
V="$V;h2h-hold|--opp ckpt --opp-ckpt $P --raw 1 --deck-set holdout --seeds 2 --ckpt $S"
V="$V;h2h-new|--opp ckpt --opp-ckpt $P --raw 1 --deck-set newcards --seeds 3 --ckpt $S"
CKPT="$P" SM_ARGS="--deal-seed0 7000" VARIANTS="$V" PROBE_NPROC=1 VAR_CAP="${VAR_CAP:-60m}" \
  bash scripts/vast_search_match.sh "$NAME"
tick card
uv run --no-sync python scripts/audit/paired_sm.py "$OUT/praw" "$OUT/student" > "$OUT/paired.txt" 2>&1 \
  || alert "paired_sm: $(tail -1 "$OUT/paired.txt")"
echo "$(date +%T) paired student - P: $(sed -n 1,4p "$OUT/paired.txt" | tr '\n' ' ')" >> "$OUT/PROGRESS"
for s in s ya; do
  uv run --no-sync python scripts/audit/principles.py --name "student-$s" --games "$OUT/student" --seat $s \
    --out "$OUT/principles-student-$s.json" > "$OUT/principles-student-$s.txt" 2>&1 || alert "principles $s failed"
done
uv run --no-sync python - "$OUT" <<'PY' >> "$OUT/PROGRESS"
import json, sys
o = sys.argv[1]
s, ya = (json.load(open(f"{o}/principles-student-{x}.json")) for x in ("s", "ya"))
print(f"student behaviour: own-T1 HT games {s['lines_with_own_t1_ht']} (gate <= 1.5 x ygo-agent "
      f"{ya['lines_with_own_t1_ht']}); interrupts {s['interrupt_opp']['rate']} vs ygo-agent {ya['interrupt_opp']['rate']}")
PY
