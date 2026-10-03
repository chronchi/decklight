#!/usr/bin/env bash
# BC recipe comparison on the SAME shards (generalisation to unseen decks):
# shards once (CPU), then train 5 variants one after another (GPU).
# Usage: bash scripts/vast_bc_recipe.sh <name> [epochs]. Results in runs/<name>/:
#   <variant>/metrics.jsonl, <variant>/best.pt, <variant>-train.txt, summary.tsv.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; EPOCHS="${2:-15}"; OUT="runs/$NAME"; mkdir -p "$OUT"
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
if [ "$Q" = "max" ]; then J=$(nproc); else J=$(( Q / P )); fi
[ "$J" -lt 2 ] && J=2
uv sync --group dev --group ml -q
uv pip install -q --reinstall torch --index-url https://download.pytorch.org/whl/cu128 > "$OUT/torch.txt" 2>&1 || true
{ nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; echo "jobs $J";
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; } > "$OUT/machine.txt" 2>&1
T0=$SECONDS
# shards from every record source (guides, claude, muse, games)
uv run --no-sync python scripts/build_bc_shards.py --out data/cache/bc_shards --jobs "$J" > "$OUT/shards.txt" 2>&1
echo "shards $((SECONDS - T0)) s" > "$OUT/timing.txt"
HOLD="kewl-tune $( [ -f decks/pool/HOLDOUT ] && tr '\n' ' ' < decks/pool/HOLDOUT )"
train_variant() {
  V="$1"; shift
  VT0=$SECONDS
  uv run --no-sync python scripts/train_bc.py --shards data/cache/bc_shards --emb data/cards/emb/qwen3-0.6b-d512.npz \
    --out "$OUT/$V" --epochs "$EPOCHS" --batch 512 --lr 5e-4 --holdout $HOLD --select-on holdout "$@" \
    > "$OUT/$V-train.txt" 2>&1
  echo "train $V $((SECONDS - VT0)) s" >> "$OUT/timing.txt"
  rm -f "$OUT/$V/last.pt"
}
train_variant base
train_variant reg --dropout 0.3 --wd 0.1 --label-smoothing 0.1
train_variant carddrop --card-dropout 0.15
train_variant reg_carddrop --dropout 0.3 --wd 0.1 --label-smoothing 0.1 --card-dropout 0.15
train_variant small --d 128 --layers 3 --dropout 0.3 --wd 0.1 --label-smoothing 0.1
# summary.tsv: variant, best val acc (epoch), best holdout acc (epoch), holdout per-deck at best holdout epoch, train s
uv run --no-sync python - "$OUT" <<'EOF' > "$OUT/summary.tsv"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print("variant\tbest_val_acc\tbest_val_epoch\tbest_holdout_acc\tbest_holdout_epoch\tholdout_per_deck\ttrain_seconds")
for v in ["base", "reg", "carddrop", "reg_carddrop", "small"]:
    m = out / v / "metrics.jsonl"
    if not m.exists():
        print(f"{v}\tNA\tNA\tNA\tNA\t{{}}\tNA")
        continue
    recs = [json.loads(line) for line in m.read_text().splitlines() if line.strip()]
    bv, be, bh, bhe, pd, ts = -1.0, -1, -1.0, -1, {}, 0.0
    for r in recs:
        ts = r.get("time", ts)
        if "val" in r and r["val"]["acc"] > bv:
            bv, be = r["val"]["acc"], r["epoch"]
        if "holdout" in r and r["holdout"]["acc"] > bh:
            bh, bhe, pd = r["holdout"]["acc"], r["epoch"], r["holdout"].get("per_deck", {})
    print(f"{v}\t{bv:.4f}\t{be}\t{bh:.4f}\t{bhe}\t{json.dumps(pd, sort_keys=True)}\t{ts:.0f}")
EOF
echo done > "$OUT/DONE"
