#!/usr/bin/env bash
# ExIt pilot students (PLAN §0a "ExIt pilot", decisions 2026-10-10 in NOTEBOOK): fine-tune base_r3 on the filtered search
# labels (data/demos/sw-xg, kinds sw-xg_own / sw-xg) at 25 / 50 / 100% of the training games, then play base_r3 and every
# student raw vs ygo-agent on the same deals (both seats). Env: BASE (default data/cache/base_r3/best.pt), GA_SEEDS (24 =
# 2,976 deals per seat), EPOCHS (6), STUDENT_ARGS (extra train_bc args).
# Usage: bash scripts/vast_exit_students.sh <name>. Do NOT run on the laptop (vast GPU only).
# Results: runs/<name>/s{25,50,100}/ (metrics.jsonl, best.pt, final.pt), runs/<name>/{base,s25,s50,s100}/games-*.jsonl.
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
T0=$SECONDS
tick() { echo "$1 $((SECONDS - T0)) s" >> "$OUT/timing.txt"; }
BASE="${BASE:-data/cache/base_r3/best.pt}"
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
J=$(nproc); [ "$Q" != max ] && J=$(( Q / P ))
uv run --no-sync python scripts/build_bc_shards.py --glob 'data/demos/sw-xg/*.jsonl' --out data/cache/bc_shards \
  --jobs "$J" > "$OUT/shards.txt" 2>&1
tick shards
grep -q "^samples: [1-9]" "$OUT/shards.txt" || { alert "no ExIt shard samples: $(tail -1 "$OUT/shards.txt")"; exit 3; }
HOLD="kewl-tune $( [ -f decks/pool/HOLDOUT ] && tr '\n' ' ' < decks/pool/HOLDOUT )"
# shellcheck disable=SC2206
ST=(--shards data/cache/bc_shards --emb data/cards/emb/qwen3-0.6b-d512.npz --holdout $HOLD --init "$BASE"
    --epochs "${EPOCHS:-6}" --lr 1e-4 --batch 128 --select-on last --kinds sw-xg sw-xg_own --require-kinds sw-xg_own
    --soft-targets gumbel --soft-norm minmax --soft-scale 3 --kl-init 0.3 ${STUDENT_ARGS:-})
V="base|--opp ygoagent --raw 1"
for F in 25 50 100; do
  uv run --no-sync python scripts/train_bc.py "${ST[@]}" --train-frac "$(python3 -c "print($F / 100)")" --out "$OUT/s$F" \
    > "$OUT/train-s$F.txt" 2>&1
  rc=$?
  rm -f "$OUT/s$F/last.pt"
  tick "train-s$F"
  grep -h "^ALERT" "$OUT/train-s$F.txt" >> "$OUT/ALERT"
  if [ $rc -ne 0 ] || [ ! -s "$OUT/s$F/best.pt" ]; then alert "train-s$F failed (exit $rc): $(tail -1 "$OUT/train-s$F.txt")"; continue; fi
  echo "s$F $(grep -c '"epoch"' "$OUT/s$F/metrics.jsonl") metric lines" >> "$OUT/PROGRESS"
  V="$V;s$F|--opp ygoagent --raw 1 --ckpt $OUT/s$F/best.pt"
done
# every policy raw on the same deals (31 ygo-agent lists x 4 of our decks x GA_SEEDS x 2 seats); paired later with
# scripts/audit/paired_gate.py runs/<name>/base runs/<name>/s<F>
CKPT="$BASE" SM_ARGS="--ga-seeds ${GA_SEEDS:-24}" VARIANTS="$V" PROBE_NPROC=1 VAR_CAP="${VAR_CAP:-60m}" \
  bash scripts/vast_search_match.sh "$NAME"
tick matches
