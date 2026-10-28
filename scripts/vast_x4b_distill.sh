#!/usr/bin/env bash
# X4b steps 3-6 (docs/plans/2026-10-28-specialists-distill.md; pre-registered NOTEBOOK 2026-10-28 "X4b"):
#   1. games (gen_selfplay, temp 1.0, opponent P argmax): each specialist pilots its 3 training lists (SPEC_GAMES
#      each); P pilots the anchor decks (ANCHOR_GAMES). Anchor decks = pool - HOLDOUT - kewl-tune - every x4b
#      related/unrelated/held-out list, so the KL-to-P anchor never sits on a deck the cards score. Opponent decks:
#      the same anchor set. Asserts: >= 95% of games written, cut < 2%, each training list >= 100 student games.
#   2. shards (kind pgame), student D and control C. D: train_bc --init P --kl-ref-map (each training list -> its specialist,
#      default P) --ce-coef 0 (pure distillation), --select-on last. Milestone: epoch 1's train KL to the
#      specialists must be below epoch 0's (D is moving toward them), else stop.
#   3. cards (raw, opponent P raw, both seats, paired with a P arm on the same deals, deal seeds 29000+):
#      unrelated (H1 + non-inferiority), related, each archetype (H2), and D vs ygo-agent on the gauntlet.
#   4. scripts/audit/x4b_distill_card.py -> runs/<name>/distill_card.{json,txt} with the gates.
#   Control C: the same games and loss with P the only teacher (--kl-ref P). Not exactly P: training runs with
#   dropout 0.1, so KL to P has a gradient. C is carded on unrelated and related (gate D - C > 0).
# Env: SPEC_DIR (data/cache/x4b_spec: <arch>/best.pt + metrics.jsonl, spec_card.json for H2), ARCHS (the 5), SPEC_GAMES (1500),
#   ANCHOR_GAMES (7500), EPOCHS (2), LR (1e-4), UNREL_SEEDS (13), REL_SEEDS (4), ARCH_SEEDS (24), GA_SEEDS (24),
#   SMOKE (1: tiny everything), VAR_CAP (40m).
# Usage: bash scripts/vast_x4b_distill.sh <name>. vast GPU only.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_x4b_distill.sh <name>}"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
tick() { echo "$(date +%T) $1" | tee -a "$OUT/PROGRESS"; }
ARCHS=${ARCHS:-branded-light-and-darkness-ritual invoked radiant-typhoon sacred-beasts atlantean-mermail}
SPEC_DIR=${SPEC_DIR:-data/cache/x4b_spec}
SPEC_GAMES=${SPEC_GAMES:-1500}; ANCHOR_GAMES=${ANCHOR_GAMES:-7500}; EPOCHS=${EPOCHS:-2}; LR=${LR:-1e-4}
UNREL_SEEDS=${UNREL_SEEDS:-13}; REL_SEEDS=${REL_SEEDS:-4}; ARCH_SEEDS=${ARCH_SEEDS:-24}; GA_SEEDS=${GA_SEEDS:-24}
SMOKE=${SMOKE:-0}; LIM=(); MIN_LIST=100
# smoke: no --limit (it keeps the first samples only, so later archetypes get 0 rows; 30k samples train fast)
if [ "$SMOKE" = 1 ]; then SPEC_GAMES=40; ANCHOR_GAMES=120; EPOCHS=2; MIN_LIST=5; fi
P=data/cache/P/best.pt

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
for a in $ARCHS; do
  for f in "$SPEC_DIR/$a/best.pt" "$SPEC_DIR/$a/metrics.jsonl"; do [ -s "$f" ] || { alert "missing $f"; exit 3; }; done
  cmp -s "$SPEC_DIR/$a/best.pt" "$P" && { alert "$SPEC_DIR/$a/best.pt is P"; exit 3; }
done

# Anchor deck set and the distillation map.
uv run --no-sync python - "$OUT" "$SPEC_DIR" $ARCHS <<'PY' > "$OUT/decks.txt" 2>&1 || { alert "deck sets: $(tail -1 "$OUT/decks.txt")"; exit 3; }
import json, shutil, sys
from pathlib import Path
out, spec, archs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
split = json.loads(Path("decks/x4b/split.json").read_text())
scored = set(split["related"]) | set(split["unrelated_fresh"]) | set(split["unrelated_holdout"]) | set(split["heldout"].values())
hold = set(Path("decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
train = {s for a in archs for s in split["train"][a]}
anchor = out / "anchor_decks"; anchor.mkdir(exist_ok=True)
n = 0
for p in sorted(Path("decks/pool").glob("*.ydk")):
    if p.stem not in hold | scored | train:
        shutil.copy(p, anchor / p.name); n += 1
m = {"default": "data/cache/P/best.pt", "decks": {s: f"{spec}/{a}/best.pt" for a in archs for s in split["train"][a]}}
(out / "kl_map.json").write_text(json.dumps(m, indent=1) + "\n")
print(f"anchor decks {n} (pool minus HOLDOUT, scored x4b lists and training lists); map {len(m['decks'])} lists")
if n < 100:
    sys.exit(f"only {n} anchor decks")
PY
tick "$(cat "$OUT/decks.txt")"

# 1. games: one gen_selfplay per (source, process); seed blocks of 10000 per process
GMB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); RMB=$(free -m | awk '/Mem/{print $2}')
NP=$(python3 -c "print(max(1, min($J, $GMB // 1500, $RMB // 3000)))")
tick "games: $NP procs per source (cores $J, gpu ${GMB}MB, ram ${RMB}MB)"
gen() { # <tag> <ckpt> <decks dir> <games> <seed0>
  local tag="$1" ck="$2" dk="$3" games="$4" s0="$5" per p pids=()
  per=$(( (games + NP - 1) / NP ))
  for ((p = 0; p < NP; p++)); do
    uv run --no-sync python scripts/gen_selfplay.py --ckpt "$ck" --out "$OUT/pgames/$tag-p$p" --games "$per" --temp 1.0 \
      --league "P=$P:1" --decks "$dk" --opp-decks "$OUT/anchor_decks" --seed $((s0 + 10000 * p)) --threads 1 \
      --device cuda > "$OUT/gen-$tag-$p.txt" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || alert "gen $tag proc $p exit $?"; done
}
k=0
for a in $ARCHS; do
  gen "$a" "$SPEC_DIR/$a/best.pt" "decks/x4b/train/$a" "$SPEC_GAMES" $((600000 + 1000000 * k)); k=$((k + 1))
  tick "games $a: $(cat "$OUT"/pgames/"$a"-p*/*.jsonl 2>/dev/null | wc -l)"
done
gen anchor "$P" "$OUT/anchor_decks" "$ANCHOR_GAMES" 6600000
tick "games anchor: $(cat "$OUT"/pgames/anchor-p*/*.jsonl 2>/dev/null | wc -l)"
uv run --no-sync python - "$OUT" "$SPEC_GAMES" "$ANCHOR_GAMES" "$MIN_LIST" $ARCHS <<'PY' >> "$OUT/PROGRESS" || { alert "games asserts: $(tail -1 "$OUT/PROGRESS")"; exit 4; }
import collections, glob, json, sys
from pathlib import Path
out, sg, ag, min_list, archs = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
split = json.loads(Path("decks/x4b/split.json").read_text())
for tag, want in [*((a, sg) for a in archs), ("anchor", ag)]:
    n = cut = win = 0
    per = collections.Counter()
    for f in glob.glob(str(out / f"pgames/{tag}-p*/*.jsonl")):
        for line in open(f):
            r = json.loads(line); n += 1
            per[r["decks"][r["student_seat"]]] += 1
            cut += r["winner"] == 3; win += r["winner"] == r["student_seat"]
    print(f"games {tag}: {n}, cut {cut}, student win rate vs P {win / max(n, 1):.3f}, decks {len(per)}")
    lists = split["train"].get(tag, [])
    low = [s for s in lists if per[s] < min_list]
    if n < 0.95 * want or cut > 0.02 * max(n, 1) or low or (lists and set(per) - set(lists)):
        sys.exit(f"ALERT games {tag}: {n}/{want}, cut {cut}, lists under {min_list}: {low}, others {sorted(set(per) - set(lists))[:3] if lists else []}")
PY

# 2. shards + student D
uv run --no-sync python scripts/build_bc_shards.py --glob "$OUT/pgames/*/*.jsonl" --out data/cache/bc_shards_x4b --jobs "$J" \
  > "$OUT/shards.txt" 2>&1 || { alert "shards failed: $(tail -1 "$OUT/shards.txt")"; exit 3; }
grep -h "^ALERT" "$OUT/shards.txt" >> "$OUT/ALERT"
tick "shards $(grep -m1 '^samples' "$OUT/shards.txt")"
HOLD="kewl-tune $(tr '\n' ' ' < decks/pool/HOLDOUT)"
# shellcheck disable=SC2086
uv run --no-sync python scripts/train_bc.py --shards data/cache/bc_shards_x4b --emb data/cards/emb/qwen3-0.6b-d512.npz \
  --out "$OUT/D" --init "$P" --epochs "$EPOCHS" --batch 512 --lr "$LR" --holdout $HOLD --select-on last \
  --kinds pgame --require-kinds pgame --kl-ref-map "$OUT/kl_map.json" --kl-ref-coef 1.0 --ce-coef 0 \
  "${LIM[@]}" > "$OUT/train.txt" 2>&1 &
TP=$!
( seen=0; while kill -0 $TP 2>/dev/null; do  # relay each epoch; milestone: specialist KL falls from epoch 0 to 1
    sleep 60; n=$(grep -c '"kl_teacher"' "$OUT/D/metrics.jsonl" 2>/dev/null); n=${n:-0}
    if [ "$n" -gt "$seen" ]; then
      echo "$(date +%T) train $(grep '"kl_teacher"' "$OUT/D/metrics.jsonl" | tail -1 | cut -c1-500)" >> "$OUT/PROGRESS"; seen=$n
      if [ "$n" = 2 ] && ! uv run --no-sync python -c "
import json, sys
r = [json.loads(x) for x in open('$OUT/D/metrics.jsonl') if '\"kl_teacher\"' in x]
spec = [k for k in r[0]['kl_teacher'] if k != 'P']
a, b = (sum(x['kl_teacher'][k] * x['teacher_rows'][k] for k in spec) / sum(x['teacher_rows'][k] for k in spec) for x in r[:2])
print(f'specialist KL epoch0 {a:.4f} epoch1 {b:.4f}')
sys.exit(0 if b < a else 1)" >> "$OUT/PROGRESS"; then
        alert "D: specialist KL did not fall from epoch 0 to 1; stopping"; kill $TP
      fi
    fi
  done ) &
wait $TP; rc=$?
grep -h "^ALERT" "$OUT/train.txt" >> "$OUT/ALERT"
[ $rc -eq 0 ] && [ -s "$OUT/D/best.pt" ] || { alert "D training failed (exit $rc): $(tail -1 "$OUT/train.txt")"; exit 4; }
rm -f "$OUT/D/last.pt"
tick "D trained"
# control C: same shards, epochs and lr; P the only teacher
# shellcheck disable=SC2086
if uv run --no-sync python scripts/train_bc.py --shards data/cache/bc_shards_x4b --emb data/cards/emb/qwen3-0.6b-d512.npz \
    --out "$OUT/C" --init "$P" --epochs "$EPOCHS" --batch 512 --lr "$LR" --holdout $HOLD --select-on last \
    --kinds pgame --require-kinds pgame --kl-ref "$P" --kl-ref-coef 1.0 --ce-coef 0 > "$OUT/train-C.txt" 2>&1 \
    && [ -s "$OUT/C/best.pt" ]; then
  rm -f "$OUT/C/last.pt"; tick "C trained: $(grep '"train_kl"' "$OUT/C/metrics.jsonl" | tail -1 | cut -c1-200)"
  CV=1
else
  alert "C training failed: $(tail -1 "$OUT/train-C.txt"); C arms skipped"; CV=0
fi
grep -h "^ALERT" "$OUT/train-C.txt" >> "$OUT/ALERT"
rm -rf data/cache/bc_shards_x4b

# 3. cards: P and D on the same deals
mkdir -p "$OUT/decks"; find decks/x4b -name '*.ydk' -exec cp {} "$OUT/decks/" \;
D="--ckpt $OUT/D/best.pt"
V="P-unrel|--pairs @decks/x4b/pairs/unrelated.pairs --seeds $UNREL_SEEDS"
V="$V;D-unrel|$D --pairs @decks/x4b/pairs/unrelated.pairs --seeds $UNREL_SEEDS"
V="$V;P-rel|--pairs @decks/x4b/pairs/related.pairs --seeds $REL_SEEDS"
V="$V;D-rel|$D --pairs @decks/x4b/pairs/related.pairs --seeds $REL_SEEDS"
if [ "$CV" = 1 ]; then
  V="$V;C-unrel|--ckpt $OUT/C/best.pt --pairs @decks/x4b/pairs/unrelated.pairs --seeds $UNREL_SEEDS"
  V="$V;C-rel|--ckpt $OUT/C/best.pt --pairs @decks/x4b/pairs/related.pairs --seeds $REL_SEEDS"
fi
for a in $ARCHS; do
  V="$V;P-$a|--pairs @decks/x4b/pairs/arch-$a.pairs --seeds $ARCH_SEEDS"
  V="$V;D-$a|$D --pairs @decks/x4b/pairs/arch-$a.pairs --seeds $ARCH_SEEDS"
done
V="$V;P-ga|--opp ygoagent --ga-seeds $GA_SEEDS;D-ga|$D --opp ygoagent --ga-seeds $GA_SEEDS"
SM="--raw 1 --opp ckpt --opp-ckpt $P --seed-per-pair 1 --deal-seed0 29000 --deck-dir $OUT/decks"
[ "$SMOKE" = 1 ] && { SM="$SM --games-limit 512"; PROBE_NPROC=${PROBE_NPROC:-2}; }
tick "card variants: ${V//;/ ; }"
PROBE_NPROC=${PROBE_NPROC:-40} CKPT="$P" SM_ARGS="$SM" VARIANTS="$V" VAR_CAP="${VAR_CAP:-40m}" \
  bash scripts/vast_search_match.sh "$NAME/card" > "$OUT/card.log" 2>&1 || { alert "card job exit $?"; }

SPEC=(); [ -f "$SPEC_DIR/spec_card.json" ] && SPEC=(--spec "$SPEC_DIR/spec_card.json")
# shellcheck disable=SC2086
if uv run --no-sync python scripts/audit/x4b_distill_card.py "$OUT/card" $ARCHS "${SPEC[@]}" > "$OUT/distill_card.txt" 2>&1; then
  tick "distill card: $(grep -h '^GATE' "$OUT/distill_card.txt" | tr '\n' ' ')"
else
  alert "x4b_distill_card failed: $(tail -1 "$OUT/distill_card.txt")"
fi
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
