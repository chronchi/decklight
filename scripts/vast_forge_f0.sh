#!/usr/bin/env bash
# Forge F0 (docs/plans/2026-10-29-deck-forge.md; pre-registered NOTEBOOK 2026-10-29 "Forge F0"):
#   (no E2 throughput sweep: X4b's metrics and NOTEBOOK 2026-10-06 17:35 already answer it, NOTEBOOK "Forge F0")
#   1. E1: one specialist per archetype in decks/forge/f0/split.json, from the release (D), FT_ITERS iterations,
#      snapshots every SNAP iterations, PAR at a time;
#   2. card (scripts/vast_search_match.sh in runs/<name>/card): raw, both seats, opponent D raw, per archetype R-<a>
#      (D) and S<it>-<a> for it in CARD_ITS, same pairs + seeds = paired; plus P-rep-invoked, the X4b specialist card's
#      P-invoked arm re-run (same deals) for the L3a determinism check (compared on the laptop);
#   3. scripts/audit/forge_curve_card.py -> runs/<name>/curve_card.{json,txt}.
# Env: ARCHS (split.json's), FT_ITERS (160), SNAP (20), CARD_ITS ("20 40 60 80 120 160"), FT_HOURS (6, a cap),
#   SHARE (0.5), ARCH_GAMES (1536: seeds per pair = ceil(ARCH_GAMES / 2 / pairs)), PAR (2), MS_IT (40), MS_MIN (0.5),
#   SMOKE (1: FT_ITERS 6, SNAP 3, CARD_ITS "3 6", 512 games per arm), VAR_CAP (25m).
# Usage: bash scripts/vast_forge_f0.sh <name>. Do NOT run on the laptop (vast GPU only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_forge_f0.sh <name>}"
OUT="runs/$NAME"
mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
tick() { echo "$(date +%T) $1" | tee -a "$OUT/PROGRESS"; }
F0=decks/forge/f0
REL=data/cache/x4b_D/best.pt
ARCHS=${ARCHS:-$(python3 -c "import json; print(' '.join(json.load(open('$F0/split.json'))['archs']))")}
FT_ITERS=${FT_ITERS:-160}; SNAP=${SNAP:-20}; CARD_ITS=${CARD_ITS:-20 40 60 80 120 160}
FT_HOURS=${FT_HOURS:-6}; SHARE=${SHARE:-0.5}; ARCH_GAMES=${ARCH_GAMES:-1536}
MS_IT=${MS_IT:-40}; MS_MIN=${MS_MIN:-0.5}
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then
  FT_ITERS=6; SNAP=3; CARD_ITS="3 6"; MS_IT=0  # SNAP must skip the 2 value-only warmup iterations
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
[ -f "$REL" ] || { alert "missing release $REL"; exit 3; }
for a in $ARCHS; do
  n=$(ls "$F0/train/$a"/*.ydk 2>/dev/null | wc -l)
  [ "$n" = 3 ] || { alert "$F0/train/$a has $n lists, not 3"; exit 3; }
  [ -f "$F0/pairs/arch-$a.pairs" ] || { alert "missing $F0/pairs/arch-$a.pairs"; exit 3; }
done

n_arch=$(echo "$ARCHS" | wc -w)
PAR=${PAR:-2}  # packing more runs per GPU gives ~1.0x (X4b A5000: 23.8 s/it alone, 47.6 s/it two at a time)
[ "$PAR" -gt "$n_arch" ] && PAR=$n_arch

# ---- 1. E1 specialists ---------------------------------------------------------------------------------------------
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
  if taskset -c "$lo-$hi" uv run --no-sync python scripts/finetune_deck.py --deck $(ls "$F0/train/$a"/*.ydk) \
      --init "$REL" --out "$dir" --iters "$FT_ITERS" --hours "$FT_HOURS" --share "$SHARE" --snap-every "$SNAP" \
      --device cuda > "$OUT/ft-$a.log" 2>&1; then
    tick "ft-$a done ($(grep -o '"iterations": [0-9]*' "$dir/FINETUNE.json"))"
  else
    alert "ft-$a failed: $(grep -h 'ALERT\|Error' "$OUT/ft-$a.log" | tail -1)"
  fi
  touch "$dir/DONE_FT"
  kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
  rm -f "$dir/rl/last.pt"  # optimizer state only; policy/, value/ and snaps/ (fp16 exports) stay
}

tick "fine-tunes from $REL: $ARCHS, $FT_ITERS it each, snapshots every $SNAP, $PAR at a time, $((J / PAR)) cores each"
slot=0
for a in $ARCHS; do
  ft_one "$a" "$slot" &
  slot=$(( (slot + 1) % PAR ))
  while [ "$(jobs -rp | wc -l)" -ge $((PAR + 1)) ]; do wait -n; done
done
for a in $ARCHS; do while [ ! -f "$OUT/ft-$a/DONE_FT" ]; do sleep 20; done; done

# ---- 2. card -------------------------------------------------------------------------------------------------------
mkdir -p "$OUT/decks"
find "$F0" decks/x4b -name '*.ydk' -exec cp {} "$OUT/decks/" \;
snap_of() { # <arch> <it> -> policy checkpoint path ("" when missing)
  local p="$OUT/ft-$1/snaps/it-$(printf %04d "$2")/policy/best.pt"
  [ "$2" = "$FT_ITERS" ] && p="$OUT/ft-$1/policy/best.pt"
  [ -f "$p" ] && echo "$p"
}
V="P-rep-invoked|--ckpt data/cache/P/best.pt --opp-ckpt data/cache/P/best.pt --deal-seed0 28000 --pairs @decks/x4b/pairs/arch-invoked.pairs --seeds 24"
OK=""
for a in $ARCHS; do
  [ -f "$OUT/ft-$a/policy/best.pt" ] || { alert "ft-$a has no policy/best.pt; its arms are skipped"; FAILED=1; continue; }
  OK="$OK $a"
  np_=$(grep -c . "$F0/pairs/arch-$a.pairs")
  sd=$(( (ARCH_GAMES + 2 * np_ - 1) / (2 * np_) ))  # >= ARCH_GAMES games per arm whatever the opponent count
  V="$V;R-$a|--pairs @$F0/pairs/arch-$a.pairs --seeds $sd"
  for it in $CARD_ITS; do
    p=$(snap_of "$a" "$it")
    if [ -z "$p" ]; then alert "ft-$a: no snapshot at it$it"; FAILED=1; continue; fi
    V="$V;S$it-$a|--ckpt $p --pairs @$F0/pairs/arch-$a.pairs --seeds $sd"
  done
done
[ -n "$OK" ] || { alert "no specialist exported; card skipped"; exit 3; }
SM="--raw 1 --opp ckpt --opp-ckpt $REL --seed-per-pair 1 --deal-seed0 32000 --deck-dir $OUT/decks"
[ "$SMOKE" = 1 ] && { SM="$SM --games-limit 512"; PROBE_NPROC=${PROBE_NPROC:-2}; }
tick "card variants: ${V//;/ ; }"
PROBE_NPROC=${PROBE_NPROC:-40} CKPT=$REL SM_ARGS="$SM" VARIANTS="$V" VAR_CAP="${VAR_CAP:-25m}" \
  bash scripts/vast_search_match.sh "$NAME/card" > "$OUT/card.log" 2>&1 || { alert "card job exit $?"; FAILED=1; }

# shellcheck disable=SC2086
if uv run --no-sync python scripts/audit/forge_curve_card.py "$OUT/card" --split "$F0/split.json" --its $CARD_ITS --archs $OK \
    > "$OUT/curve_card.txt" 2>&1; then
  tick "curve card: $(grep -h '^BUDGET\|^pooled' "$OUT/curve_card.txt" | tr '\n' ' ')"
else
  alert "forge_curve_card failed: $(tail -1 "$OUT/curve_card.txt")"; FAILED=1
fi
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
[ $FAILED = 0 ]
