#!/usr/bin/env bash
# Per-deck fine-tune pilot: fine-tune P + value_p2 on each deck in DECKS (scripts/finetune_deck.py),
# then a per-deck paired card raw and searched (scripts/deck_card.py) vs ygo-agent gauntlet decks.
# Pre-registered: NOTEBOOK 2026-10-22 per-deck fine-tune pilot.
# Env: DECKS ("BlueEyes Branded"), FT_HOURS (0.75), SHARE (0.5), RAW_SEEDS (8), SEARCH_SEEDS (4),
#   SMOKE (0), SMOKE_ITERS (3), NP (empty = auto), GB_PER_PROC (1.5, GPU GB per card process),
#   RAM_PER_PROC (3, GB), MS_IT (20), MS_MIN (0.45), DRY (1 = echo every command, run nothing).
# Usage: bash scripts/vast_ft_pilot.sh <name>. Do NOT run on the laptop (vast GPU only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_ft_pilot.sh <name>}"
OUT="runs/$NAME"
mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
tick() { echo "$(date +%T) $1" | tee -a "$OUT/PROGRESS"; }
DECKS=${DECKS:-BlueEyes Branded}
FT_HOURS=${FT_HOURS:-0.75}; SHARE=${SHARE:-0.5}
RAW_SEEDS=${RAW_SEEDS:-8}; SEARCH_SEEDS=${SEARCH_SEEDS:-4}
SMOKE=${SMOKE:-0}; SMOKE_ITERS=${SMOKE_ITERS:-3}
NP=${NP:-}; GB_PER_PROC=${GB_PER_PROC:-1.5}; RAM_PER_PROC=${RAM_PER_PROC:-3}
MS_IT=${MS_IT:-20}; MS_MIN=${MS_MIN:-0.45}
DRY=${DRY:-0}
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

  # NP sizing (once): NP = max(1, min(J, floor(gpu*0.85/GB_PER_PROC), floor(ram*0.8/RAM_PER_PROC))),
  # capped at the smaller card's deal count so no proc gets 0 deals.
  if [ -z "$NP" ]; then
    GTOT_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    RTOT_GB=$(free -g | awk '/Mem/{print $2}')
    RAW_DEALS=$((30 * 2 * RAW_SEEDS)); SEARCH_DEALS=$((30 * 2 * SEARCH_SEEDS))
    if [ "$SMOKE" = 1 ]; then RAW_DEALS=$((30 * 2 * 1)); SEARCH_DEALS=$((30 * 2 * 1)); fi
    if [ "$SEARCH_DEALS" -lt "$RAW_DEALS" ]; then CAP_DEALS=$SEARCH_DEALS; else CAP_DEALS=$RAW_DEALS; fi
    NP=$(python3 -c "print(max(1, min($J, int($GTOT_MB / 1024 * 0.85 / $GB_PER_PROC), int($RTOT_GB * 0.8 / $RAM_PER_PROC), $CAP_DEALS)))")
    {
      echo "cores $J, gpu ${GTOT_MB}MB, ram ${RTOT_GB}GB; per-proc ${GB_PER_PROC}GB/${RAM_PER_PROC}GB -> NP=$NP (deal cap $CAP_DEALS)"
    } | tee "$OUT/np.txt"
  else
    echo "NP=$NP (env)" | tee "$OUT/np.txt"
  fi
else
  NP=${NP:-4}
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
    if [ $rc -eq 2 ] && [ $ms_done = 0 ]; then
      ms_done=1
      echo "$out" | grep -h "^MILESTONE_FAIL" >> "$OUT/ALERT" 2>/dev/null || true
      alert "$tag milestone it$MS_IT mean focus_wr_vs_bc below $MS_MIN (run continues)"
    fi
    [ -f "$dir/DONE_FT" ] && break
  done
}

card_progress() { # <tag> <dir>: append "PROGRESS <tag> base=.. ft=.. diff=.. ci=[lo,hi] n=.." from card.json.
  local tag="$1" dir="$2"
  uv run --no-sync python - "$tag" "$dir/card.json" >> "$OUT/PROGRESS" 2>"$dir/progress.txt" <<'EOF' || return 1
import json, sys
from pathlib import Path
tag, path = sys.argv[1], sys.argv[2]
card = json.loads(Path(path).read_text())
arms = card["arms"]
first = card["diff_vs_first"]["first"]
others = [k for k in card["diff_vs_first"] if k != "first"]
ft = others[0] if others else first
d = card["diff_vs_first"][ft] if ft != first else {"diff_pp": 0.0, "ci_pp": [0.0, 0.0], "n": arms[first]["n"]}
print(f"PROGRESS {tag} base={arms[first]['win_rate']} ft={arms[ft]['win_rate']} "
      f"diff={d['diff_pp']} ci=[{d['ci_pp'][0]},{d['ci_pp'][1]}] n={d['n']}")
EOF
}

run_card() { # <deck> <search01> <seeds> <tag>: NP procs + summarize; returns 1 when the searched card must be skipped.
  local D="$1" search="$2" seeds="$3" tag="$4"
  local dir="$OUT/$tag"
  mkdir -p "$dir"
  local deck="decks/gauntlet/$D.ydk"
  local base_arm="base=data/cache/P/best.pt:data/cache/value_p2/best.pt"
  local ft_arm="ft=$OUT/ft-$D/policy/best.pt:$OUT/ft-$D/value/best.pt"
  local p pid rc procs_failed=0
  if [ "$DRY" = 1 ]; then
    for ((p = 0; p < NP; p++)); do
      runv scripts/deck_card.py --deck "$deck" --arm "$base_arm" --arm "$ft_arm" \
        --search "$search" --seeds "$seeds" --device cuda --out "$dir" --nproc "$NP" --proc "$p"
    done
    runv scripts/deck_card.py --deck "$deck" --arm "$base_arm" --arm "$ft_arm" \
      --search "$search" --seeds "$seeds" --device cuda --out "$dir" --nproc "$NP" --proc 0 --summarize
    return 0
  fi
  local pids=()
  for ((p = 0; p < NP; p++)); do
    # one torch thread per card proc (NP procs share the cores); the trainer keeps its defaults
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 runv scripts/deck_card.py --deck "$deck" --arm "$base_arm" --arm "$ft_arm" \
      --search "$search" --seeds "$seeds" --device cuda --out "$dir" \
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
  if ! runv scripts/deck_card.py --deck "$deck" --arm "$base_arm" --arm "$ft_arm" \
      --search "$search" --seeds "$seeds" --device cuda --out "$dir" \
      --nproc "$NP" --proc 0 --summarize > "$dir/summarize.txt" 2>&1; then
    alert "$tag: summarize failed: $(tail -1 "$dir/summarize.txt")"
    FAILED=1
    return 0
  fi
  if ! card_progress "$tag" "$dir"; then
    alert "$tag: PROGRESS line failed: $(tail -1 "$dir/progress.txt")"
    FAILED=1
  fi
  return 0
}

for D in $DECKS; do
  FT_EXTRA=()
  if [ "$SMOKE" = 1 ]; then FT_EXTRA=(--iters "$SMOKE_ITERS"); fi
  if [ "$DRY" = 1 ]; then
    runv scripts/finetune_deck.py --deck "decks/gauntlet/$D.ydk" --out "$OUT/ft-$D" \
      --hours "$FT_HOURS" --share "$SHARE" --device cuda "${FT_EXTRA[@]}"
  else
    if [ ! -f "decks/gauntlet/$D.ydk" ]; then
      alert "missing decks/gauntlet/$D.ydk; deck skipped"
      FAILED=1
      continue
    fi
    rm -f "$OUT/ft-$D/DONE_FT"
    watch_ft "$OUT/ft-$D" "ft-$D" &
    WPID=$!
    if runv scripts/finetune_deck.py --deck "decks/gauntlet/$D.ydk" --out "$OUT/ft-$D" \
        --hours "$FT_HOURS" --share "$SHARE" --device cuda "${FT_EXTRA[@]}" > "$OUT/ft-$D.log" 2>&1; then
      tick "ft-$D done"
    else
      alert "ft-$D failed: $(tail -1 "$OUT/ft-$D.log")"
      FAILED=1
      touch "$OUT/ft-$D/DONE_FT"
      kill "$WPID" 2>/dev/null || true
      wait "$WPID" 2>/dev/null || true
      continue  # a failed stage skips this deck's cards
    fi
    touch "$OUT/ft-$D/DONE_FT"
    kill "$WPID" 2>/dev/null || true
    wait "$WPID" 2>/dev/null || true
    # train_rl's optimizer state only: drop last.pt when it is > 200 MB; never touch policy/ or value/.
    if [ -f "$OUT/ft-$D/rl/last.pt" ] && [ "$(stat -c%s "$OUT/ft-$D/rl/last.pt")" -gt 209715200 ]; then
      rm -f "$OUT/ft-$D/rl/last.pt"
    fi
  fi
  RAW_S=$RAW_SEEDS; SEARCH_S=$SEARCH_SEEDS
  if [ "$SMOKE" = 1 ]; then RAW_S=1; SEARCH_S=1; fi
  if ! run_card "$D" 0 "$RAW_S" "card-raw-$D"; then
    alert "card-raw-$D failed; searched card skipped"
    FAILED=1
    continue
  fi
  if [ "$DRY" != 1 ] && [ ! -f "$OUT/card-raw-$D/card.json" ]; then
    alert "card-raw-$D has no card.json; searched card skipped"
    FAILED=1
    continue
  fi
  run_card "$D" 1 "$SEARCH_S" "card-search-$D" || FAILED=1
  if [ "$DRY" != 1 ] && [ ! -f "$OUT/card-search-$D/card.json" ]; then
    alert "card-search-$D has no card.json"
    FAILED=1
  fi
  tick "deck $D done"
done

if [ "$DRY" = 1 ]; then
  echo '{"dry": true}' > "$OUT/pilot.json"
  cat "$OUT/pilot.json"
  exit 0
fi

# End: collect every card.json + FINETUNE.json into pilot.json.
if ! uv run --no-sync python - "$OUT" "$DECKS" <<'EOF'; then
import json, sys
from pathlib import Path
out, decks = Path(sys.argv[1]), sys.argv[2].split()
pilot: dict = {"decks": {}}
for d in decks:
    entry: dict = {}
    for k, p in (("raw", out / f"card-raw-{d}/card.json"),
                 ("search", out / f"card-search-{d}/card.json"),
                 ("finetune", out / f"ft-{d}/FINETUNE.json")):
        entry[k] = json.loads((p).read_text()) if p.is_file() else None
    pilot["decks"][d] = entry
(out / "pilot.json").write_text(json.dumps(pilot, indent=1))
print(json.dumps(pilot, indent=1)[:2000])
EOF
  alert "pilot.json failed"
  FAILED=1
fi
if [ "$DRY" != 1 ]; then
  kill "$USAGE_PID" 2>/dev/null || true
fi
[ $FAILED = 0 ]
