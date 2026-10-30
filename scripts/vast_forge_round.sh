#!/usr/bin/env bash
# Forge round (docs/plans/2026-10-29-deck-forge.md, "Pipeline" steps 2-5): one vast job runs a whole round:
#   1. train a specialist per SPLIT archetype from the release (FT_ITERS, PAR at a time), pack each
#      (scripts/forge.py pack from the release) and validate; pre-built PACKS_IN packs are copied in too;
#   2. verify every pack with the group-sequential test (scripts/audit/forge_seqtest.py): up to K looks on
#      fresh deals per look, ACCEPT/REJECT per pack, the survivors stay LIVE;
#   3. distill the accepted packs into NEXT_NAME (teacher games per pack + the anchor, shards,
#      scripts/train_bc.py pure distillation with --select-on last);
#   4. card the release (fixed n: new archetypes, old X4b archetypes, reserve, ygo-agent gauntlet);
#   5. scripts/audit/forge_release_card.py -> runs/<name>/release_card.{json,txt} with the gates.
# Budget: F2 ~= 5 h on a 3090 (train ~= 1.6 h, verify ~= 0.6 h, distill ~= 1.3 h, card ~= 1.4 h).
# Env: REL (data/cache/x4b_D/best.pt), REL_NAME (r0), NEXT_NAME (r1), SPLIT (decks/forge/f2/split.json),
#   ARCHS (SPLIT's archs), PACKS_IN (data/forge/packs), FT_ITERS (40), PAR (2), SHARE (0.5), FT_HOURS (3, a cap),
#   OPPS (decks/x4b/opponents.txt), LOOK_GAMES (256), K (6), MARGIN (0.03), VERIFY_SEED0 (41000),
#   TEACH_GAMES (1500), ANCHOR_GAMES (7500), EPOCHS (2), LR (1e-4), CARD_SEED0 (43000), NEW_GAMES (1536),
#   OLD_ARCHS (the 5 X4b archetypes), OLD_GAMES (768), RES_PAIRS (decks/forge/f2/pairs/reserve.pairs),
#   RES_GAMES (3072), GA_SEEDS (24), VAR_CAP (40m),
#   SMOKE (1: FT_ITERS 4, LOOK_GAMES 64, K 2, TEACH_GAMES 40, ANCHOR_GAMES 120, EPOCHS 2,
#   every card arm --games-limit 256, PROBE_NPROC 2, the train_bc milestone only alerts).
# Usage: bash scripts/vast_forge_round.sh <name>. Do NOT run on the laptop (vast GPU only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_forge_round.sh <name>}"
OUT="runs/$NAME"
mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
tick() { echo "$(date +%T) $1" | tee -a "$OUT/PROGRESS"; }
REL=${REL:-data/cache/x4b_D/best.pt}
REL_NAME=${REL_NAME:-r0}
NEXT_NAME=${NEXT_NAME:-r1}
SPLIT=${SPLIT:-decks/forge/f2/split.json}
SDIR=$(dirname "$SPLIT")
ARCHS=${ARCHS:-$(python3 -c "import json; print(' '.join(json.load(open('$SPLIT'))['archs']))")}
PACKS_IN=${PACKS_IN:-data/forge/packs}
FT_ITERS=${FT_ITERS:-40}; PAR=${PAR:-2}; SHARE=${SHARE:-0.5}; FT_HOURS=${FT_HOURS:-3}
OPPS=${OPPS:-decks/x4b/opponents.txt}
LOOK_GAMES=${LOOK_GAMES:-256}; K=${K:-6}; MARGIN=${MARGIN:-0.03}; VERIFY_SEED0=${VERIFY_SEED0:-41000}
TEACH_GAMES=${TEACH_GAMES:-1500}; ANCHOR_GAMES=${ANCHOR_GAMES:-7500}; EPOCHS=${EPOCHS:-2}; LR=${LR:-1e-4}
CARD_SEED0=${CARD_SEED0:-43000}
NEW_GAMES=${NEW_GAMES:-1536}
OLD_ARCHS=${OLD_ARCHS:-branded-light-and-darkness-ritual invoked radiant-typhoon sacred-beasts atlantean-mermail}
OLD_GAMES=${OLD_GAMES:-768}
RES_PAIRS=${RES_PAIRS:-decks/forge/f2/pairs/reserve.pairs}
RES_GAMES=${RES_GAMES:-3072}; GA_SEEDS=${GA_SEEDS:-24}; VAR_CAP=${VAR_CAP:-40m}
MS_IT=20; MS_MIN=0.5
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then
  FT_ITERS=4; LOOK_GAMES=64; K=2; TEACH_GAMES=40; ANCHOR_GAMES=120; EPOCHS=2
  PROBE_NPROC=${PROBE_NPROC:-2}
fi
MIN_LIST=100; [ "$SMOKE" = 1 ] && MIN_LIST=5
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
[ -f "$REL" ] || { alert "missing release $REL"; exit 3; }
[ -f "$(dirname "$REL")/metrics.jsonl" ] || { alert "missing $(dirname "$REL")/metrics.jsonl"; exit 3; }
for a in $ARCHS; do
  n=$(ls "$SDIR/train/$a"/*.ydk 2>/dev/null | wc -l)
  [ "$n" = 3 ] || { alert "$SDIR/train/$a has $n lists, not 3"; exit 3; }
done
for d in "$PACKS_IN"/*/; do
  uv run --no-sync python scripts/forge.py validate "$d" --release "$REL" > "$OUT/validate-$(basename "$d").txt" 2>&1 \
    || { alert "pre-built pack $d invalid: $(grep -h '^ERROR' "$OUT/validate-$(basename "$d").txt" | head -1)"; exit 3; }
done

n_arch=$(echo "$ARCHS" | wc -w)
[ "$PAR" -gt "$n_arch" ] && PAR=$n_arch

# ---- 1. train specialists ----------------------------------------------------------------------------------
watch_ft() { # <ftdir> <tag>: metrics.jsonl -> PROGRESS every 30 s + the MS_IT milestone (alert only)
  local dir="$1" tag="$2" seen=0 ms_done=0 m="$1/rl/metrics.jsonl" out maxit rc
  while true; do
    sleep 30
    [ -f "$dir/DONE_FT" ] && break
    [ -f "$m" ] || continue
    out=$(uv run --no-sync python scripts/audit/ft_progress.py "$m" --tag "$tag" \
      --seen "$seen" --ms-it "$MS_IT" --ms-min "$MS_MIN" 2>&1)
    rc=$?
    echo "$out" | grep -h "^PROGRESS\|^MILESTONE " >> "$OUT/PROGRESS" 2>/dev/null || true
    maxit=$(echo "$out" | grep -oh "it=[0-9]*" | grep -oh "[0-9]*" | sort -n | tail -1)
    [ -n "${maxit:-}" ] && seen=$maxit
    if [ $rc -eq 2 ] && [ $ms_done = 0 ]; then
      ms_done=1
      alert "$tag milestone it$MS_IT mean focus_wr_vs_bc below $MS_MIN (run continues)"
    fi
  done
}

ft_one() { # <arch> <slot>
  local a="$1" slot="$2" dir="$OUT/ft-$1" c=$((J / PAR)) wpid
  local lo=$((slot * c)) hi=$((slot * c + c - 1))
  mkdir -p "$dir"; rm -f "$dir/DONE_FT"
  watch_ft "$dir" "ft-$a" & wpid=$!
  # shellcheck disable=SC2046
  if taskset -c "$lo-$hi" uv run --no-sync python scripts/finetune_deck.py --deck $(ls "$SDIR/train/$a"/*.ydk) \
      --init "$REL" --out "$dir" --iters "$FT_ITERS" --hours "$FT_HOURS" --share "$SHARE" \
      --device cuda > "$OUT/ft-$a.log" 2>&1; then
    tick "ft-$a done ($(grep -o '"iterations": [0-9]*' "$dir/FINETUNE.json"))"
  else
    alert "ft-$a failed: $(grep -h 'ALERT\|Error' "$OUT/ft-$a.log" | tail -1)"
  fi
  touch "$dir/DONE_FT"
  kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
  rm -f "$dir/rl/last.pt"  # optimizer state only; policy/ and value/ (fp16 exports) stay
}

tick "fine-tunes from $REL: $ARCHS, $FT_ITERS it each, $PAR at a time, $((J / PAR)) cores each"
# each run takes a free core slot (slot-<s>.busy markers); round-robin by launch order put two runs on the same cores
# while the other slot idled (F2 2026-10-30: lunalight and trains both on cores 0-12)
rm -f "$OUT"/slot-*.busy
for a in $ARCHS; do
  while true; do
    slot=""
    for ((s = 0; s < PAR; s++)); do [ -f "$OUT/slot-$s.busy" ] || { slot=$s; break; }; done
    [ -n "$slot" ] && break
    sleep 10
  done
  touch "$OUT/slot-$slot.busy"
  ( ft_one "$a" "$slot"; rm -f "$OUT/slot-$slot.busy" ) &
done
for a in $ARCHS; do while [ ! -f "$OUT/ft-$a/DONE_FT" ]; do sleep 20; done; done

mkdir -p "$OUT/packs"
TRAINED=""
for a in $ARCHS; do
  [ -f "$OUT/ft-$a/policy/best.pt" ] || { alert "ft-$a has no policy/best.pt; it is dropped"; continue; }
  held=$(python3 -c "import json; print(json.load(open('$SPLIT'))['heldout']['$a'])")
  # shellcheck disable=SC2086
  if ! uv run --no-sync python scripts/forge.py pack --ft "$OUT/ft-$a" --release "$REL" \
      --release-name "$REL_NAME" --archetype "$a" --train-decks $SDIR/train/$a/*.ydk \
      --heldout-decks "$SDIR/heldout/$held.ydk" --out "$OUT/packs/$a" --sources "$SPLIT" \
      > "$OUT/pack-$a.txt" 2>&1; then
    alert "$a pack failed: $(tail -1 "$OUT/pack-$a.txt")"; continue
  fi
  if ! uv run --no-sync python scripts/forge.py validate "$OUT/packs/$a" --release "$REL" \
      >> "$OUT/pack-$a.txt" 2>&1; then
    alert "$a pack invalid: $(grep -h '^ERROR' "$OUT/pack-$a.txt" | head -1)"; rm -rf "$OUT/packs/$a"; continue
  fi
  rm -f "$OUT/ft-$a/rl/last.pt"
  TRAINED="$TRAINED $a"
  tick "$a packed: $(grep -m1 '^forge pack' "$OUT/pack-$a.txt")"
done
for d in "$PACKS_IN"/*/; do
  cp -r "$d" "$OUT/packs/$(basename "$d")"
done
PACKS=$(ls "$OUT/packs")
[ -n "$PACKS" ] || { alert "no packs (no pre-built packs and every fine-tune failed)"; exit 3; }
tick "packs: $PACKS (trained:$TRAINED)"

# ---- 2. verify (sequential) --------------------------------------------------------------------------------
mkdir -p "$OUT/verify" "$OUT/decks"
: > "$OUT/verify/LIVE"
for p in $PACKS; do
  if ! uv run --no-sync python scripts/forge_round_prep.py pairs --pack "$OUT/packs/$p" \
      --opponents "$OPPS" --out "$OUT/verify/$p.pairs" > "$OUT/verify/$p.pairs.log" 2>&1; then
    alert "$p pairs failed: $(tail -1 "$OUT/verify/$p.pairs.log")"; FAILED=1; continue
  fi
  pairs_p=$(grep -c . "$OUT/verify/$p.pairs")
  s_p=$(uv run --no-sync python scripts/forge_round_prep.py seeds --pairs "$OUT/verify/$p.pairs" --games "$LOOK_GAMES")
  [ $((pairs_p * s_p)) -lt 1000 ] || { alert "$p: pairs*seeds $((pairs_p * s_p)) >= 1000"; exit 3; }
  echo "$s_p" > "$OUT/verify/$p.seeds"
  echo $((2 * pairs_p * s_p)) > "$OUT/verify/$p.block"
  echo "$p" >> "$OUT/verify/LIVE"
  tick "$p verify: $(cat "$OUT/verify/$p.pairs.log"), seeds $s_p block $(cat "$OUT/verify/$p.block")"
done
find "$OUT/packs" -name '*.ydk' -exec cp {} "$OUT/decks/" \;
for s in $(cut -d: -f2 "$OUT"/verify/*.pairs | sort -u); do
  [ -f "decks/pool/$s.ydk" ] || { alert "opponent $s has no decks/pool/$s.ydk"; exit 3; }
  cp "decks/pool/$s.ydk" "$OUT/decks/"
done
NPROC_FROM=""
for ((k = 1; k <= K; k++)); do
  [ -s "$OUT/verify/LIVE" ] || break
  V=""
  for a in $(cat "$OUT/verify/LIVE"); do
    s_p=$(cat "$OUT/verify/$a.seeds")
    V="$V;R-$a|--pairs @$OUT/verify/$a.pairs --seeds $s_p"
    V="$V;S-$a|--ckpt $OUT/packs/$a/policy/best.pt --pairs @$OUT/verify/$a.pairs --seeds $s_p"
  done
  V="${V#;}"
  SM="--raw 1 --opp ckpt --opp-ckpt $REL --seed-per-pair 1 --deal-seed0 $((VERIFY_SEED0 + 1000 * (k - 1))) --deck-dir $OUT/decks"
  if [ $k = 1 ]; then
    PROBE_NPROC=${PROBE_NPROC:-40} CKPT=$REL SM_ARGS="$SM" VARIANTS="$V" VAR_CAP="$VAR_CAP" \
      bash scripts/vast_search_match.sh "$NAME/verify-L$k" > "$OUT/verify-L$k.log" 2>&1 \
      || { alert "verify look $k job exit $?"; FAILED=1; }
    NPROC_FROM=$(grep -oh '[0-9][0-9]* procs' "$OUT/verify-L1/sizing.txt" 2>/dev/null | grep -oh '[0-9][0-9]*' | head -1)
    [ -n "${NPROC_FROM:-}" ] || { alert "verify look 1 has no process count in sizing.txt"; exit 3; }
  else
    NPROC=$NPROC_FROM CKPT=$REL SM_ARGS="$SM" VARIANTS="$V" VAR_CAP="$VAR_CAP" \
      bash scripts/vast_search_match.sh "$NAME/verify-L$k" > "$OUT/verify-L$k.log" 2>&1 \
      || { alert "verify look $k job exit $?"; FAILED=1; }
  fi
  NEWLIVE=""
  for a in $(cat "$OUT/verify/LIVE"); do
    L_p=$(cat "$OUT/verify/$a.block")
    RDIRS=""; SDIRS=""
    for ((j = 1; j <= k; j++)); do
      RDIRS="$RDIRS,$OUT/verify-L$j/R-$a"; SDIRS="$SDIRS,$OUT/verify-L$j/S-$a"
    done
    RDIRS="${RDIRS#,}"; SDIRS="${SDIRS#,}"
    # shellcheck disable=SC2086
    if sout=$(uv run --no-sync python scripts/audit/forge_seqtest.py --release $RDIRS --specialist $SDIRS \
        --look "$L_p" --K "$K" --margin "$MARGIN" --json "$OUT/verify/$a.json" 2>&1); then
      dec=$(python3 -c "import json; print(json.load(open('$OUT/verify/$a.json'))['decision'])")
      tick "$a look $k: $(echo "$sout" | grep -h '^DECISION')"
      case "$dec" in ACCEPT|REJECT) ;; *) NEWLIVE="$NEWLIVE $a";; esac
    else
      alert "$a look $k seqtest incomplete: $(echo "$sout" | tail -1)"
      rm -f "$OUT/verify/$a.json"
      echo "$a" >> "$OUT/verify/REJECTED"
    fi
  done
  for a in $NEWLIVE; do echo "$a"; done > "$OUT/verify/LIVE.new"
  # LIVE = NEWLIVE (ACCEPT/REJECT/incomplete packs leave)
  : > "$OUT/verify/LIVE"
  for a in $NEWLIVE; do echo "$a" >> "$OUT/verify/LIVE"; done
  rm -f "$OUT/verify/LIVE.new"
done
[ -s "$OUT/verify/LIVE" ] && { alert "undecided after $K looks: $(cat "$OUT/verify/LIVE" | xargs)"; FAILED=1; : > "$OUT/verify/LIVE"; }
# shellcheck disable=SC2086
uv run --no-sync python - "$OUT" $PACKS <<'PY' >> "$OUT/PROGRESS" 2>&1 || { alert "verify.json failed"; exit 3; }
import json, sys
from pathlib import Path
out, packs = Path(sys.argv[1]), sys.argv[2:]
res = {}
for a in packs:
    jf = out / "verify" / f"{a}.json"
    pack = json.loads((out / "packs" / a / "pack.json").read_text())
    if jf.is_file():
        r = json.loads(jf.read_text())
        res[a] = {"pack_id": pack["pack_id"], "decision": r["decision"], "look": r["look"],
                  "n": r["n"], "mean_pp": r["mean_pp"], "rci_pp": r["rci_pp"]}
    else:
        res[a] = {"pack_id": pack["pack_id"], "decision": "REJECT", "look": 0, "n": 0,
                  "mean_pp": 0.0, "rci_pp": [0.0, 0.0]}
(out / "verify.json").write_text(json.dumps(res, indent=1) + "\n")
print("verify " + json.dumps({a: v["decision"] for a, v in res.items()}))
PY
ACCEPTED=$(python3 -c "import json; print(' '.join(a for a, v in json.load(open('$OUT/verify.json')).items() if v['decision'] == 'ACCEPT'))")
if [ -z "$ACCEPTED" ]; then
  alert "no pack accepted; nothing to distill"
  kill "$USAGE_PID" 2>/dev/null || true
  echo done > "$OUT/DONE"
  exit 0
fi
tick "accepted: $ACCEPTED"

# ---- 3. distill --------------------------------------------------------------------------------------------
if ! uv run --no-sync python scripts/forge_round_prep.py anchor --packs "$OUT"/packs/*/ --out "$OUT/anchor_decks" \
    > "$OUT/anchor.txt" 2>&1; then
  alert "anchor: $(tail -1 "$OUT/anchor.txt")"; exit 4
fi
ACCPACKS=""; for a in $ACCEPTED; do ACCPACKS="$ACCPACKS $OUT/packs/$a"; done
# shellcheck disable=SC2086
uv run --no-sync python scripts/forge_round_prep.py klmap --packs $ACCPACKS --default "$REL" \
  --out "$OUT/kl_map.json" > "$OUT/klmap.txt" 2>&1 || { alert "klmap: $(tail -1 "$OUT/klmap.txt")"; exit 4; }
tick "$(cat "$OUT/anchor.txt") $(cat "$OUT/klmap.txt")"
GMB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); RMB=$(free -m | awk '/Mem/{print $2}')
# per-process GPU memory from the verify probe (one policy), x2 for gen_selfplay's league opponent; 1500 MB when
# unmeasured. A fixed 1500 MB left 10 of 26 cores idle in F2 (probe 530 MB).
PGMB=$(grep -oh 'probe gpu [0-9]*MB' "$OUT/verify-L1/sizing.txt" 2>/dev/null | grep -oh '[0-9]*' | head -1)
GPP=$(( ${PGMB:-750} * 2 ))
NP=$(python3 -c "print(max(1, min($J, $GMB // $GPP, $RMB // 3000)))")
tick "games: $NP procs per source (cores $J, gpu ${GMB}MB, ram ${RMB}MB)"
gen() { # <tag> <ckpt> <decks dir> <games> <seed0>
  local tag="$1" ck="$2" dk="$3" games="$4" s0="$5" per p pids=()
  per=$(( (games + NP - 1) / NP ))
  mkdir -p "$OUT/pgames"
  for ((p = 0; p < NP; p++)); do
    uv run --no-sync python scripts/gen_selfplay.py --ckpt "$ck" --out "$OUT/pgames/$tag-p$p" --games "$per" --temp 1.0 \
      --league "R=$REL:1" --decks "$dk" --opp-decks "$OUT/anchor_decks" --seed $((s0 + 10000 * p)) --threads 1 \
      --device cuda > "$OUT/gen-$tag-$p.txt" 2>&1 &
    pids+=($!)
  done
  # watchdog: once half the workers are done, kill any still running after 3x the median finish time + 120 s
  # (F2 2026-10-30: one kewl-tune worker hung 16 min in an engine loop with no decision, so the decision cap never fired)
  local t0 done_t=() alive n med lim
  t0=$(date +%s)
  while true; do
    alive=0; done_t=()
    for p in "${pids[@]}"; do
      if kill -0 "$p" 2>/dev/null; then alive=$((alive + 1)); else done_t+=("$(cat "$OUT/pgames/.done-$p" 2>/dev/null || { date +%s > "$OUT/pgames/.done-$p"; cat "$OUT/pgames/.done-$p"; })"); fi
    done
    [ "$alive" = 0 ] && break
    n=${#done_t[@]}
    if [ "$n" -ge $(( (${#pids[@]} + 1) / 2 )) ]; then
      med=$(printf '%s\n' "${done_t[@]}" | sort -n | awk -v t0="$t0" '{a[NR]=$1-t0} END{print a[int((NR+1)/2)]}')
      lim=$(( 3 * med + 120 ))
      if [ $(( $(date +%s) - t0 )) -gt "$lim" ]; then
        for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && { alert "gen $tag: worker $p still running after ${lim}s (3x median + 120); killed"; pkill -P "$p"; kill "$p"; }; done
      fi
    fi
    sleep 10
  done
  for p in "${pids[@]}"; do wait "$p" || alert "gen $tag proc $p exit $?"; rm -f "$OUT/pgames/.done-$p"; done
}
i=0
for a in $ACCEPTED; do
  gen "$a" "$OUT/packs/$a/policy/best.pt" "$OUT/packs/$a/decks/train" "$TEACH_GAMES" $((700000 + 1000000 * i)); i=$((i + 1))
  tick "games $a: $(cat "$OUT"/pgames/"$a"-p*/*.jsonl 2>/dev/null | wc -l)"
done
gen anchor "$REL" "$OUT/anchor_decks" "$ANCHOR_GAMES" 7700000
tick "games anchor: $(cat "$OUT"/pgames/anchor-p*/*.jsonl 2>/dev/null | wc -l)"
WANTS=""; for a in $ACCEPTED; do WANTS="$WANTS $a:$TEACH_GAMES"; done
# shellcheck disable=SC2086
uv run --no-sync python - "$OUT" "$MIN_LIST" $WANTS "anchor:$ANCHOR_GAMES" <<'PY' >> "$OUT/PROGRESS" 2>&1
import collections, glob, json, sys
from pathlib import Path
out, min_list = Path(sys.argv[1]), int(sys.argv[2])
fails = []
for spec in sys.argv[3:]:
    tag, want = spec.split(":"); want = int(want)
    pack = out / "packs" / tag / "pack.json"
    lists = [d["stem"] for d in json.loads(pack.read_text())["decks"] if d["role"] == "train"] if pack.is_file() else []
    n = cut = win = 0
    per = collections.Counter()
    for f in glob.glob(str(out / f"pgames/{tag}-p*/*.jsonl")):
        for line in open(f):
            r = json.loads(line); n += 1
            per[r["decks"][r["student_seat"]]] += 1
            cut += r["winner"] == 3; win += r["winner"] == r["student_seat"]
    print(f"games {tag}: {n}, cut {cut}, student win rate vs R {win / max(n, 1):.3f}, decks {len(per)}")
    low = [s for s in lists if per[s] < min_list]
    bad = (n < 0.95 * want or cut > 0.02 * max(n, 1) or low
           or (lists and sorted(set(per) - set(lists))))
    if bad:
        print(f"FAIL {tag}: {n}/{want}, cut {cut}, lists under {min_list}: {low}, "
              f"others {sorted(set(per) - set(lists))[:3] if lists else []}")
        fails.append(tag)
Path(out / "games_fail.txt").write_text("\n".join(fails) + "\n")
PY
[ -f "$OUT/games_fail.txt" ] || { alert "games asserts crashed: $(tail -1 "$OUT/PROGRESS")"; exit 4; }
if grep -qx anchor "$OUT/games_fail.txt"; then
  alert "anchor games failed asserts"; exit 4
fi
for a in $(cat "$OUT/games_fail.txt"); do
  [ "$a" = anchor ] && continue
  [ -n "$a" ] || continue
  alert "games $a failed asserts; $a dropped from the distillation map"
  # shellcheck disable=SC2086
  ACCEPTED=$(echo $ACCEPTED | tr ' ' '\n' | grep -vx "$a" | xargs)
done
[ -n "$ACCEPTED" ] || { alert "every pack's games failed asserts"; exit 4; }
ACCPACKS=""; for a in $ACCEPTED; do ACCPACKS="$ACCPACKS $OUT/packs/$a"; done
# shellcheck disable=SC2086
uv run --no-sync python scripts/forge_round_prep.py klmap --packs $ACCPACKS --default "$REL" \
  --out "$OUT/kl_map.json" >> "$OUT/klmap.txt" 2>&1 || { alert "klmap rebuild: $(tail -1 "$OUT/klmap.txt")"; exit 4; }
uv run --no-sync python scripts/build_bc_shards.py --glob "$OUT/pgames/*/*.jsonl" --out data/cache/bc_shards_forge --jobs "$J" \
  > "$OUT/shards.txt" 2>&1 || { alert "shards failed: $(tail -1 "$OUT/shards.txt")"; exit 3; }
grep -h "^ALERT" "$OUT/shards.txt" >> "$OUT/ALERT"
tick "shards $(grep -m1 '^samples' "$OUT/shards.txt")"
HOLD="kewl-tune $(tr '\n' ' ' < decks/pool/HOLDOUT)"
# shellcheck disable=SC2086
uv run --no-sync python scripts/train_bc.py --shards data/cache/bc_shards_forge --emb data/cards/emb/qwen3-0.6b-d512.npz \
  --out "$OUT/$NEXT_NAME" --init "$REL" --epochs "$EPOCHS" --batch 512 --lr "$LR" --holdout $HOLD --select-on last \
  --kinds pgame --require-kinds pgame --kl-ref-map "$OUT/kl_map.json" --kl-ref-coef 1.0 --ce-coef 0 \
  > "$OUT/train.txt" 2>&1 &
TP=$!
( seen=0; while kill -0 $TP 2>/dev/null; do  # each epoch; milestone: pack-teacher KL falls from epoch 0 to 1
    sleep 60; n=$(grep -c '"kl_teacher"' "$OUT/$NEXT_NAME/metrics.jsonl" 2>/dev/null); n=${n:-0}
    if [ "$n" -gt "$seen" ]; then
      echo "$(date +%T) train $(grep '"kl_teacher"' "$OUT/$NEXT_NAME/metrics.jsonl" | tail -1 | cut -c1-500)" >> "$OUT/PROGRESS"; seen=$n
      if [ "$n" = 2 ] && ! uv run --no-sync python - "$OUT/$NEXT_NAME/metrics.jsonl" "$OUT/kl_map.json" <<'PY' >> "$OUT/PROGRESS"; then
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if '"kl_teacher"' in x]
klmap = json.load(open(sys.argv[2]))
from pathlib import Path as _P
_d = _P(klmap["default"]).parent
default_lab = _d.parent.name if _d.name == "policy" else _d.name  # train_bc's teacher label rule
spec = [k for k in rows[0]["kl_teacher"] if k != default_lab]
a, b = (sum(x["kl_teacher"][k] * x["teacher_rows"][k] for k in spec) / sum(x["teacher_rows"][k] for k in spec) for x in rows[:2])
print(f"pack-teacher KL epoch0 {a:.4f} epoch1 {b:.4f}")
sys.exit(0 if b < a else 1)
PY
        if [ "$SMOKE" = 1 ]; then
          alert "$NEXT_NAME: pack-teacher KL did not fall from epoch 0 to 1 (smoke: run continues)"
        else
          alert "$NEXT_NAME: pack-teacher KL did not fall from epoch 0 to 1; stopping"; kill $TP
        fi
      fi
    fi
  done ) &
wait $TP; rc=$?
grep -h "^ALERT" "$OUT/train.txt" >> "$OUT/ALERT"
[ $rc -eq 0 ] && [ -s "$OUT/$NEXT_NAME/best.pt" ] || { alert "$NEXT_NAME training failed (exit $rc): $(tail -1 "$OUT/train.txt")"; exit 4; }
rm -f "$OUT/$NEXT_NAME/last.pt"
tick "$NEXT_NAME trained"
rm -rf data/cache/bc_shards_forge
cmp -s "$OUT/$NEXT_NAME/best.pt" "$REL" && { alert "$NEXT_NAME/best.pt is $REL"; exit 4; }

# ---- 4. release card (fixed n) ---------------------------------------------------------------------------------
find decks/x4b decks/forge -name '*.ydk' -exec cp {} "$OUT/decks/" \;
N="--ckpt $OUT/$NEXT_NAME/best.pt"
V=""
for a in $ACCEPTED; do
  s=$(uv run --no-sync python scripts/forge_round_prep.py seeds --pairs "$OUT/verify/$a.pairs" --games "$NEW_GAMES")
  V="$V;R-$a|--pairs @$OUT/verify/$a.pairs --seeds $s"
  V="$V;N-$a|$N --pairs @$OUT/verify/$a.pairs --seeds $s"
done
# shellcheck disable=SC2086
for a in $OLD_ARCHS; do
  s=$(uv run --no-sync python scripts/forge_round_prep.py seeds --pairs "decks/x4b/pairs/arch-$a.pairs" --games "$OLD_GAMES")
  V="$V;R-$a|--pairs @decks/x4b/pairs/arch-$a.pairs --seeds $s"
  V="$V;N-$a|$N --pairs @decks/x4b/pairs/arch-$a.pairs --seeds $s"
done
s=$(uv run --no-sync python scripts/forge_round_prep.py seeds --pairs "$RES_PAIRS" --games "$RES_GAMES")
V="$V;R-res|--pairs @$RES_PAIRS --seeds $s"
V="$V;N-res|$N --pairs @$RES_PAIRS --seeds $s"
V="$V;R-ga|--opp ygoagent --ga-seeds $GA_SEEDS;N-ga|$N --opp ygoagent --ga-seeds $GA_SEEDS"
V="${V#;}"
SM="--raw 1 --opp ckpt --opp-ckpt $REL --seed-per-pair 1 --deal-seed0 $CARD_SEED0 --deck-dir $OUT/decks"
[ "$SMOKE" = 1 ] && SM="$SM --games-limit 256"
tick "card variants: ${V//;/ ; }"
NPROC=$NPROC_FROM CKPT=$REL SM_ARGS="$SM" VARIANTS="$V" VAR_CAP="$VAR_CAP" \
  bash scripts/vast_search_match.sh "$NAME/card" > "$OUT/card.log" 2>&1 || { alert "card job exit $?"; FAILED=1; }

# ---- 5. release card audit ---------------------------------------------------------------------------------------
# shellcheck disable=SC2086
if uv run --no-sync python scripts/audit/forge_release_card.py "$OUT/card" --verify "$OUT/verify.json" \
    --new $ACCEPTED --old $OLD_ARCHS --name "$NEXT_NAME" > "$OUT/release_card.txt" 2>&1; then
  tick "release card: $(grep -h '^GATE\|^RELEASE' "$OUT/release_card.txt" | tr '\n' ' ')"
else
  alert "forge_release_card failed: $(tail -1 "$OUT/release_card.txt")"; FAILED=1
fi
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
[ $FAILED = 0 ]
