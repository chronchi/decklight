#!/usr/bin/env bash
# X4b steps 1-2 (docs/plans/2026-10-28-specialists-distill.md; pre-registered NOTEBOOK 2026-10-28 "X4b"):
#   1. one specialist per archetype: finetune_deck on its 3 training lists (decks/x4b/train/<arch>/*.ydk), PAR at a
#      time on the one GPU, each pinned to its share of the cores;
#   2. specialist card (scripts/vast_search_match.sh in runs/<name>/card): raw, both seats, opponent P raw, every pair
#      on its own deal seeds. Arms per archetype: P-<a> and S-<a> on decks/x4b/pairs/arch-<a>.pairs (3 train lists +
#      the held-out list), P-unrel and S-<a>-unrel on unrelated.pairs. Same pairs file + seeds = paired games.
#   3. scripts/audit/x4b_spec_card.py: paired diffs + the stop rule -> runs/<name>/spec_card.{json,txt}.
# Env: ARCHS (the 5), FT_ITERS (160: the ft-curve's best snapshot), FT_HOURS (3, a cap only), SHARE (0.5), PAR (2),
#   ARCH_SEEDS (24: 32 pairs x 24 x 2 seats = 1,536 games/arm), UNREL_SEEDS (5: 152 x 5 x 2 = 1,520),
#   MS_IT (40), MS_MIN (0.5), SMOKE (1: 4 iters, 512 games per arm), VAR_CAP (25m).
# Usage: bash scripts/vast_x4b_spec.sh <name>. Do NOT run on the laptop (vast GPU only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_x4b_spec.sh <name>}"
OUT="runs/$NAME"
mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
tick() { echo "$(date +%T) $1" | tee -a "$OUT/PROGRESS"; }
ARCHS=${ARCHS:-branded-light-and-darkness-ritual invoked radiant-typhoon sacred-beasts atlantean-mermail}
FT_ITERS=${FT_ITERS:-160}; FT_HOURS=${FT_HOURS:-3}; SHARE=${SHARE:-0.5}; PAR=${PAR:-2}
ARCH_SEEDS=${ARCH_SEEDS:-24}; UNREL_SEEDS=${UNREL_SEEDS:-5}
MS_IT=${MS_IT:-40}; MS_MIN=${MS_MIN:-0.5}
SMOKE=${SMOKE:-0}
# smoke: 4 iterations (the first 2 are value-only warmup, train_rl --value-warmup 2)
if [ "$SMOKE" = 1 ]; then FT_ITERS=4; MS_IT=0; fi
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
for a in $ARCHS; do
  n=$(ls decks/x4b/train/"$a"/*.ydk 2>/dev/null | wc -l)
  [ "$n" = 3 ] || { alert "decks/x4b/train/$a has $n lists, not 3"; exit 3; }
  [ -f "decks/x4b/pairs/arch-$a.pairs" ] || { alert "missing decks/x4b/pairs/arch-$a.pairs"; exit 3; }
done
[ -f decks/x4b/pairs/unrelated.pairs ] || { alert "missing decks/x4b/pairs/unrelated.pairs"; exit 3; }

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

ft_one() { # <arch> <slot>: the specialist on its 3 lists, pinned to cores slot*C..slot*C+C-1
  local a="$1" slot="$2" dir="$OUT/ft-$1" c=$((J / PAR)) wpid
  local lo=$((slot * c)) hi=$((slot * c + c - 1))
  mkdir -p "$dir"; rm -f "$dir/DONE_FT"
  watch_ft "$dir" "ft-$a" & wpid=$!
  # shellcheck disable=SC2046
  if taskset -c "$lo-$hi" uv run --no-sync python scripts/finetune_deck.py --deck $(ls decks/x4b/train/"$a"/*.ydk) \
      --out "$dir" --iters "$FT_ITERS" --hours "$FT_HOURS" --share "$SHARE" --device cuda > "$OUT/ft-$a.log" 2>&1; then
    tick "ft-$a done ($(grep -o '"iterations": [0-9]*' "$dir/FINETUNE.json"))"
  else
    alert "ft-$a failed: $(grep -h 'ALERT\|Error' "$OUT/ft-$a.log" | tail -1)"
  fi
  touch "$dir/DONE_FT"
  kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
  # train_rl's optimizer state only; policy/ and value/ (fp16 exports) stay
  rm -f "$dir/rl/last.pt"
}

tick "fine-tunes: $ARCHS, $FT_ITERS it each, $PAR at a time, $((J / PAR)) cores each"
slot=0
for a in $ARCHS; do
  ft_one "$a" "$slot" &
  slot=$(( (slot + 1) % PAR ))
  # PAR in flight: wait for one to finish before starting the next (its slot is the one that frees first only
  # when finish order = start order; overlapping pinning for a while is harmless)
  while [ "$(jobs -rp | wc -l)" -ge $((PAR + 1)) ]; do wait -n; done
done
wait_ft() { for a in $ARCHS; do while [ ! -f "$OUT/ft-$a/DONE_FT" ]; do sleep 20; done; done; }
wait_ft

# Card: every archetype whose specialist exported, plus P on the same pairs.
mkdir -p "$OUT/decks"
find decks/x4b -name '*.ydk' -exec cp {} "$OUT/decks/" \;
V=""
OK=""
for a in $ARCHS; do
  if [ ! -f "$OUT/ft-$a/policy/best.pt" ]; then
    alert "ft-$a has no policy/best.pt; its arms are skipped"; FAILED=1; continue
  fi
  OK="$OK $a"
  s="--ckpt $OUT/ft-$a/policy/best.pt"
  V="$V;P-$a|--pairs @decks/x4b/pairs/arch-$a.pairs --seeds $ARCH_SEEDS"
  V="$V;S-$a|$s --pairs @decks/x4b/pairs/arch-$a.pairs --seeds $ARCH_SEEDS"
  V="$V;S-$a-unrel|$s --pairs @decks/x4b/pairs/unrelated.pairs --seeds $UNREL_SEEDS"
done
[ -n "$OK" ] || { alert "no specialist exported; card skipped"; exit 3; }
V="P-unrel|--pairs @decks/x4b/pairs/unrelated.pairs --seeds $UNREL_SEEDS$V"
SM="--raw 1 --opp ckpt --opp-ckpt data/cache/P/best.pt --seed-per-pair 1 --deal-seed0 28000 --deck-dir $OUT/decks"
# smoke: 512 games per arm; the probe (2 virtual procs) gets 256, enough to read its GPU memory
[ "$SMOKE" = 1 ] && { SM="$SM --games-limit 512"; PROBE_NPROC=${PROBE_NPROC:-2}; }
tick "card variants: ${V//;/ ; }"
PROBE_NPROC=${PROBE_NPROC:-40} CKPT=data/cache/P/best.pt SM_ARGS="$SM" VARIANTS="$V" VAR_CAP="${VAR_CAP:-25m}" \
  bash scripts/vast_search_match.sh "$NAME/card" > "$OUT/card.log" 2>&1 || { alert "card job exit $?"; FAILED=1; }
# the launcher relays card/ALERT and card/PROGRESS itself (vast.relay_alerts reads nested files)

# shellcheck disable=SC2086
if uv run --no-sync python scripts/audit/x4b_spec_card.py "$OUT/card" $OK > "$OUT/spec_card.txt" 2>&1; then
  tick "spec card: $(grep -h '^STOP_RULE' "$OUT/spec_card.txt")"
else
  alert "x4b_spec_card failed: $(tail -1 "$OUT/spec_card.txt")"; FAILED=1
fi
kill "$USAGE_PID" 2>/dev/null || true
echo done > "$OUT/DONE"
[ $FAILED = 0 ]
