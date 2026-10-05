#!/usr/bin/env bash
# Gold-game BC experiment (STATUS step 2, 2026-10-04): does training on the hand-played gold games help?
# Base = the best previous BC checkpoint, shipped in data/cache/bc_base/ (best.pt + metrics.jsonl).
# Variants, all with the same split (holdout decks + 25% of gold games in val) and best.pt picked by val gold accuracy:
#   zero/     base evaluated as is (epoch -1 record only)
#   ft-gold/  base fine-tuned on gold samples only
#   ft-mix/   base fine-tuned on everything, gold repeated
#   mix/      from scratch, recipe v2 + gold repeated (SKIP_SCRATCH=1 skips it)
# then playtests the base and the variant with the best val gold accuracy on the same decks/seeds.
# Usage: bash scripts/vast_bc_gold.sh <name> [smoke]. smoke = gold shards only, 1 epoch, no scratch run, tiny playtest
# (ship --data data/demos for a real playtest: it samples decks that have verified lines; with only data/demos/gold it
# finds none and the match section fails, as in bcg-smoke-20261004-2142).
# Results in runs/<name>/ (pulled back by the launcher): <variant>/metrics.jsonl + best.pt, train-*.txt, playtest-*.txt,
# summary.txt.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; MODE="${2:-full}"; OUT="runs/$NAME"; mkdir -p "$OUT"
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
if [ "$Q" = "max" ]; then J=$(nproc); else J=$(( Q / P )); fi
[ "$J" -lt 2 ] && J=2
uv sync --group dev --group ml -q
uv pip install -q --reinstall torch --index-url https://download.pytorch.org/whl/cu128 > "$OUT/torch.txt" 2>&1 || true
{ nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; echo "jobs $J";
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; } > "$OUT/machine.txt" 2>&1
T0=$SECONDS
tick() { echo "$1 $((SECONDS - T0)) s" >> "$OUT/timing.txt"; }
BASE=data/cache/bc_base/best.pt
if [ "$MODE" = "smoke" ]; then
  ONLY=(--only gold); EP_FT=1; EP_MIX=1; SKIP_SCRATCH=1; PT=(--decks 2 --hands 1 --seeds 1 --narrate 0 --samples 0); PT_CAP=8m
else
  ONLY=(); EP_FT=10; EP_MIX=3; SKIP_SCRATCH="${SKIP_SCRATCH:-0}"; PT=(--decks 10 --hands 6 --seeds 2 --narrate 2 --samples 8); PT_CAP=25m
fi
uv run --no-sync python scripts/build_bc_shards.py "${ONLY[@]}" --out data/cache/bc_shards --jobs "$J" > "$OUT/shards.txt" 2>&1
tick shards
HOLD="kewl-tune $( [ -f decks/pool/HOLDOUT ] && tr '\n' ' ' < decks/pool/HOLDOUT )"
COMMON=(--shards data/cache/bc_shards --emb data/cards/emb/qwen3-0.6b-d512.npz --batch 512 --holdout $HOLD
        --val-frac-kind gold=0.25 gold_pass=0.25 --select-on val:gold)
train() {  # <variant> <args...>
  local v="$1"; shift
  uv run --no-sync python scripts/train_bc.py "${COMMON[@]}" --out "$OUT/$v" "$@" > "$OUT/train-$v.txt" 2>&1
  rm -f "$OUT/$v/last.pt"; tick "train-$v"
}
train zero --init "$BASE" --epochs 0
train ft-gold --init "$BASE" --epochs "$EP_FT" --lr 1e-4 --batch 256 --kinds gold gold_pass
train ft-mix --init "$BASE" --epochs "$EP_MIX" --lr 2e-4 \
  --kind-frac line_expanded=0.2 game_plan=1 game_interrupt=1 gold=20 gold_pass=10
[ "$SKIP_SCRATCH" = "1" ] || train mix --epochs 8 --lr 5e-4 --merge-last 4 \
  --kind-frac line_expanded=0.2 game_plan=2 game_interrupt=2 gold=20 gold_pass=10
# summary: per variant, best record by val gold acc (zero = its epoch -1 record)
uv run --no-sync python - "$OUT" > "$OUT/summary.txt" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
best_v, best_s = "zero", -1.0
for v in ("zero", "ft-gold", "ft-mix", "mix"):
    m = out / v / "metrics.jsonl"
    if not m.exists():
        continue
    recs = [r for r in map(json.loads, filter(str.strip, m.read_text().splitlines())) if "val" in r]
    if not recs:
        continue
    r = max(recs, key=lambda r: r["val"]["per_kind"].get("gold", -1))
    g = r["val"]["per_kind"].get("gold", -1)
    hk = r.get("holdout", {}).get("per_kind", {})
    print(f"{v:8s} at {r.get('epoch', 'merged')}: val gold {g:.4f} gold_pass {r['val']['per_kind'].get('gold_pass', -1):.4f} "
          f"| holdout gold {hk.get('gold', -1):.4f} all {r.get('holdout', {}).get('acc', -1):.4f} | val all {r['val']['acc']:.4f}")
    if v != "zero" and g > best_s:
        best_v, best_s = v, g
print("playtest", best_v)
PY
BEST=$(tail -1 "$OUT/summary.txt" | awk '{print $2}')
tick "summary (best $BEST)"
for v in zero "$BEST"; do
  [ "$v" = zero ] && ck="$BASE" || ck="$OUT/$v/best.pt"
  [ -f "$ck" ] || continue
  # playtest writes to <ckpt>/../../playtest and reads the architecture from metrics.jsonl next to the checkpoint
  d="$OUT/pt-$v/ck"; mkdir -p "$d"; cp "$ck" "$d/best.pt"; cp "$OUT/$v/metrics.jsonl" "$d/"
  timeout "$PT_CAP" uv run --no-sync python scripts/playtest.py --ckpt "$d/best.pt" --device cuda --threads "$J" \
    --seed 0 "${PT[@]}" > "$OUT/playtest-$v.txt" 2>&1
  rm -f "$d/best.pt"; tick "playtest-$v"
done
echo done > "$OUT/DONE"
