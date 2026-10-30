#!/usr/bin/env bash
# Forge F3: cheaper distillation (pre-registered NOTEBOOK 2026-10-30 "Forge F3"). Reuses F2's 11 accepted packs and
# its distill games, staged on the laptop in data/forge/f3/ (packs/, anchor_decks/, pairs/, kl_map.json, tgames/ =
# F2's gen workers p0-p5 of every pack and of the anchor: ~560 teacher games per pack + 2,814 anchor games).
# Two students, both from r0 (REL) with F2's recipe (train_bc pure KL to the pack teachers, 2 epochs, --select-on last):
#   T: the teacher games in tgames/ (the pack policy pilots its decks): 3x fewer games than F2's r1;
#   D: DAgger, iteration 0: r0 pilots the pack decks on the SAME deals (F2's gen seeds of workers p0-p5, 94 games each),
#      rows labelled by the pack teacher through kl_map.json; the anchor games are T's.
# Card (raw, opponent r0 raw), paired with arms already on the laptop:
#   new: X-<a> on data/forge/f3/pairs/<a>.pairs, deal seed 43000, same seeds as F2's card (R-<a> = r0, N-<a> = r1);
#   reserve: X-res, deal seed 43000 (F2's R-res / N-res);
#   old: X-<a> on the 5 X4b archetypes, deal seed 47000, 24 seeds per pair (the confirmation card's R-/N- arms).
# Analysis on the laptop: scripts/audit/forge_f3_card.py.
# Env: REL (data/cache/x4b_D/best.pt), F3 (data/forge/f3), ARMS ("T D"), EPOCHS (2), LR (1e-4), D_GAMES (94 per
#   worker), D_WORKERS (6), NEW_GAMES (1536), RES_GAMES (3072), OLD_SEEDS (24), VAR_CAP (40m),
#   SMOKE (1: D_GAMES 4, 2 packs' D games only, tgames cut to 8 games per file, card --games-limit 128).
# Usage: bash scripts/vast_forge_f3.sh <name>. Do NOT run on the laptop (vast GPU only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_forge_f3.sh <name>}"
OUT="runs/$NAME"
mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
tick() { echo "$(date +%T) $1" | tee -a "$OUT/PROGRESS"; }
REL=${REL:-data/cache/x4b_D/best.pt}
F3=${F3:-data/forge/f3}
ARMS=${ARMS:-T D}
EPOCHS=${EPOCHS:-2}; LR=${LR:-1e-4}
D_GAMES=${D_GAMES:-94}; D_WORKERS=${D_WORKERS:-6}
NEW_GAMES=${NEW_GAMES:-1536}; RES_GAMES=${RES_GAMES:-3072}; OLD_SEEDS=${OLD_SEEDS:-24}; VAR_CAP=${VAR_CAP:-40m}
RES_PAIRS=decks/forge/f2/pairs/reserve.pairs
OLD_ARCHS="branded-light-and-darkness-ritual invoked radiant-typhoon sacred-beasts atlantean-mermail"
PACKS=$(ls "$F3/packs")
SMOKE=${SMOKE:-0}
D_PACKS="$PACKS"
if [ "$SMOKE" = 1 ]; then
  D_GAMES=4; D_PACKS=$(echo "$PACKS" | head -2 | xargs); PROBE_NPROC=${PROBE_NPROC:-2}
fi
FAILED=0

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
for f in "$REL" "$F3/kl_map.json" "$RES_PAIRS"; do [ -f "$f" ] || { alert "missing $f"; exit 3; }; done
[ "$(echo "$PACKS" | wc -w)" = 11 ] || { alert "$F3/packs has $(echo "$PACKS" | wc -w) packs, not 11"; exit 3; }
for a in $PACKS; do
  [ -f "$F3/packs/$a/policy/best.pt" ] && [ -f "$F3/pairs/$a.pairs" ] && [ -d "$F3/tgames/$a-p0" ] \
    || { alert "$a: pack, pairs or tgames missing"; exit 3; }
done

# ---- 1. games: T = the staged teacher games; D = r0 on the same deals ------------------------------------------------
mkdir -p "$OUT/games/T" "$OUT/games/D"
cp -r "$F3"/tgames/* "$OUT/games/T/"
if [ "$SMOKE" = 1 ]; then  # 8 games per deck file keep the smoke's shards and training small
  for f in "$OUT"/games/T/*/*.jsonl; do head -8 "$f" > "$f.tmp" && mv "$f.tmp" "$f"; done
fi
cp -r "$OUT"/games/T/anchor-p* "$OUT/games/D/"
GMB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); RMB=$(free -m | awk '/Mem/{print $2}')
NP=$(python3 -c "print(max(1, min($J, $GMB // 1500, $RMB // 3000)))")  # gen_selfplay: student + league opponent
UNITS="$OUT/dunits.txt"; : > "$UNITS"
i=0
for a in $PACKS; do  # F2's gen seeds: 700000 + 1e6 * (index in the sorted accepted list) + 10000 * worker
  if echo " $D_PACKS " | grep -q " $a "; then
    for ((p = 0; p < D_WORKERS; p++)); do echo "$a $p $((700000 + 1000000 * i + 10000 * p))" >> "$UNITS"; done
  fi
  i=$((i + 1))
done
tick "D games: $(wc -l < "$UNITS") units of $D_GAMES games, $NP at a time (cores $J, gpu ${GMB}MB, ram ${RMB}MB)"
export REL F3 OUT D_GAMES
# a unit is capped at 30 min (F2: one worker hung 16 min in an engine loop; a normal 94-game unit takes 2-7 min)
xargs -P "$NP" -L 1 bash -c 'timeout 30m uv run --no-sync python scripts/gen_selfplay.py --ckpt "$REL" \
    --out "$OUT/games/D/$0-p$1" --games "$D_GAMES" --temp 1.0 --league "R=$REL:1" --decks "$F3/packs/$0/decks/train" \
    --opp-decks "$F3/anchor_decks" --seed "$2" --threads 1 --device cuda > "$OUT/gen-$0-$1.txt" 2>&1 \
    || echo "ALERT gen D $0 p$1 exit $?" >> "$OUT/ALERT"' < "$UNITS"
tick "D games done: $(cat "$OUT"/games/D/*/*.jsonl 2>/dev/null | wc -l) rows (anchor included)"
# invariants: D's deals are T's (same game_id -> same decks, seed and student seat), D is not T's play (choices differ),
# (smoke: the 4-game schedule is not the 94-game one, so deals are not compared), >= 95% of the units' games present, no pack deck under 100 games (smoke: 1)
uv run --no-sync python - "$OUT" "$D_GAMES" "$D_WORKERS" "$SMOKE" $D_PACKS <<'PY' >> "$OUT/PROGRESS" 2>&1
import collections, glob, json, sys
from pathlib import Path
out, dg, dw, smoke, packs = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5:]
def load(arm, tag):
    g = {}
    for f in glob.glob(str(out / f"games/{arm}/{tag}-p*/*.jsonl")):
        for line in open(f):
            r = json.loads(line); g[r["game_id"]] = r
    return g
fails = []
for a in packs:
    t, d = load("T", a), load("D", a)
    both = set(t) & set(d)
    same_deal = sum(t[k]["decks"] == d[k]["decks"] and t[k]["student_seat"] == d[k]["student_seat"]
                    and t[k]["config"].get("seed") == d[k]["config"].get("seed") for k in both)
    same_play = sum(t[k]["choices"] == d[k]["choices"] for k in both)
    per = collections.Counter(r["decks"][r["student_seat"]] for r in d.values())
    win = sum(r["winner"] == r["student_seat"] for r in d.values()) / max(len(d), 1)
    cut = sum(r["winner"] == 3 for r in d.values())
    print(f"D {a}: {len(d)} games, paired with T {len(both)}, same deal {same_deal}, same play {same_play}, "
          f"cut {cut}, r0 win rate vs r0 {win:.3f}, per deck {dict(per)}")
    want = dg * dw
    if (len(d) < 0.95 * want or (not smoke and len(both) < 0.9 * min(len(d), len(t))) or (not smoke and same_deal < len(both))
            or same_play > 0.5 * max(len(both), 1) or cut > 0.02 * max(len(d), 1)
            or min(per.values(), default=0) < (1 if smoke else 100)):
        fails.append(a)
print("D asserts " + ("FAIL " + " ".join(fails) if fails else "OK"))
Path(out / "dgames_fail.txt").write_text(" ".join(fails) + "\n")
PY
[ -f "$OUT/dgames_fail.txt" ] || { alert "D asserts crashed: $(tail -1 "$OUT/PROGRESS")"; exit 4; }
[ -n "$(tr -d ' \n' < "$OUT/dgames_fail.txt")" ] && { alert "D games failed asserts: $(cat "$OUT/dgames_fail.txt")"; exit 4; }

# ---- 2. distill each arm (shards, then train_bc; the two trainings run together) -----------------------------------
HOLD="kewl-tune $(tr '\n' ' ' < decks/pool/HOLDOUT)"
for x in $ARMS; do
  uv run --no-sync python scripts/build_bc_shards.py --glob "$OUT/games/$x/*/*.jsonl" --out "data/cache/bc_shards_f3_$x" \
    --jobs "$J" > "$OUT/shards-$x.txt" 2>&1 || { alert "shards $x failed: $(tail -1 "$OUT/shards-$x.txt")"; exit 3; }
  grep -h "^ALERT" "$OUT/shards-$x.txt" >> "$OUT/ALERT"
  tick "shards $x $(grep -m1 '^samples' "$OUT/shards-$x.txt")"
done
TPIDS=""
for x in $ARMS; do
  # shellcheck disable=SC2086
  uv run --no-sync python scripts/train_bc.py --shards "data/cache/bc_shards_f3_$x" --emb data/cards/emb/qwen3-0.6b-d512.npz \
    --out "$OUT/$x" --init "$REL" --epochs "$EPOCHS" --batch 512 --lr "$LR" --holdout $HOLD --select-on last \
    --kinds pgame --require-kinds pgame --kl-ref-map "$F3/kl_map.json" --kl-ref-coef 1.0 --ce-coef 0 \
    --threads $((J / 2)) > "$OUT/train-$x.txt" 2>&1 &
  TPIDS="$TPIDS $!"
done
# milestone per arm at epoch 1: the pack-teacher KL falls from epoch 0 to 1 (F2's rule)
( seen=""; while true; do
    sleep 60; alive=0
    for p in $TPIDS; do kill -0 "$p" 2>/dev/null && alive=1; done
    for x in $ARMS; do
      n=$(grep -c '"kl_teacher"' "$OUT/$x/metrics.jsonl" 2>/dev/null); n=${n:-0}
      echo " $seen " | grep -q " $x:$n " && continue
      seen="$seen $x:$n"
      [ "$n" -gt 0 ] && echo "$(date +%T) train $x $(grep '"kl_teacher"' "$OUT/$x/metrics.jsonl" | tail -1 | cut -c1-400)" >> "$OUT/PROGRESS"
      if [ "$n" = 2 ] && ! uv run --no-sync python - "$OUT/$x/metrics.jsonl" <<'PY' >> "$OUT/PROGRESS"; then
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if '"kl_teacher"' in x]
spec = [k for k in rows[0]["kl_teacher"] if k != "x4b_D"]
a, b = (sum(x["kl_teacher"][k] * x["teacher_rows"][k] for k in spec) / sum(x["teacher_rows"][k] for k in spec) for x in rows[:2])
print(f"{sys.argv[1]}: pack-teacher KL epoch0 {a:.4f} epoch1 {b:.4f}")
sys.exit(0 if b < a else 1)
PY
        alert "$x: pack-teacher KL did not fall from epoch 0 to 1"
      fi
    done
    [ $alive = 0 ] && break
  done ) &
MPID=$!
CARD_ARMS=""
for p in $TPIDS; do wait "$p"; done
for x in $ARMS; do
  grep -h "^ALERT" "$OUT/train-$x.txt" >> "$OUT/ALERT"
  if [ -s "$OUT/$x/best.pt" ] && ! cmp -s "$OUT/$x/best.pt" "$REL"; then
    rm -f "$OUT/$x/last.pt"; CARD_ARMS="$CARD_ARMS $x"
    tick "$x trained: $(grep -h 'samples\|n_train' "$OUT/train-$x.txt" | head -1 | cut -c1-120)"
  else
    alert "$x training failed or best.pt is $REL: $(tail -1 "$OUT/train-$x.txt")"; FAILED=1
  fi
done
kill "$MPID" 2>/dev/null
rm -rf data/cache/bc_shards_f3_*
[ -n "$CARD_ARMS" ] || { alert "no arm trained; card skipped"; exit 4; }

# ---- 3. card: the new arms only, on the deals of F2's card (new, reserve) and of the confirmation card (old) --------
mkdir -p "$OUT/decks"
find decks/x4b decks/forge -name '*.ydk' -exec cp {} "$OUT/decks/" \;
V=""
for x in $CARD_ARMS; do
  C="--ckpt $OUT/$x/best.pt"
  for a in $PACKS; do
    s=$(uv run --no-sync python scripts/forge_round_prep.py seeds --pairs "$F3/pairs/$a.pairs" --games "$NEW_GAMES")
    V="$V;$x-$a|$C --pairs @$F3/pairs/$a.pairs --seeds $s"
  done
  s=$(uv run --no-sync python scripts/forge_round_prep.py seeds --pairs "$RES_PAIRS" --games "$RES_GAMES")
  V="$V;$x-res|$C --pairs @$RES_PAIRS --seeds $s"
  for a in $OLD_ARCHS; do
    V="$V;$x-$a|$C --pairs @decks/x4b/pairs/arch-$a.pairs --seeds $OLD_SEEDS --deal-seed0 47000"
  done
done
V="${V#;}"
SM="--raw 1 --opp ckpt --opp-ckpt $REL --seed-per-pair 1 --deal-seed0 43000 --deck-dir $OUT/decks"
[ "$SMOKE" = 1 ] && SM="$SM --games-limit 128"
tick "card variants: ${V//;/ ; }"
CKPT=$REL SM_ARGS="$SM" VARIANTS="$V" VAR_CAP="$VAR_CAP" PROBE_NPROC=${PROBE_NPROC:-40} \
  bash scripts/vast_search_match.sh "$NAME/card" > "$OUT/card.log" 2>&1 || { alert "card job exit $?"; FAILED=1; }
n_ok=$(ls -d "$OUT"/card/*/ 2>/dev/null | wc -l)
tick "card done: $n_ok arm dirs"
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
[ $FAILED = 0 ]
