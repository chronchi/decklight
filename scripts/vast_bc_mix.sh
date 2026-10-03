#!/usr/bin/env bash
# BC data-mix experiment: which data teaches what (same recipe, different --kinds).
# Shards once (all sources), then train 4 variants one after another (GPU).
# Usage: bash scripts/vast_bc_mix.sh <name> [epochs=15]. Results in runs/<name>/:
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
train_variant lines --kinds line line_improved
train_variant lines_exp --kinds line line_improved line_expanded
train_variant games --kinds game_plan game_interrupt
train_variant all
# summary.tsv: variant, n_train, best holdout acc (epoch), holdout per_kind at that epoch,
# val per_kind at the best-val epoch, train s
uv run --no-sync python - "$OUT" <<'EOF' > "$OUT/summary.tsv"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print("variant\tn_train\tbest_holdout_acc\tbest_holdout_epoch\tholdout_per_kind\tbest_val_epoch\tval_per_kind\ttrain_seconds")
for v in ["lines", "lines_exp", "games", "all"]:
    m = out / v / "metrics.jsonl"
    if not m.exists():
        print(f"{v}\tNA\tNA\tNA\t{{}}\tNA\t{{}}\tNA")
        continue
    recs = [json.loads(line) for line in m.read_text().splitlines() if line.strip()]
    n_train = recs[0].get("n_train", "NA") if recs else "NA"
    ts = recs[-1].get("time", 0.0) if recs else 0.0
    bh, bhe, hpk, bve, vpk = -1.0, -1, {}, -1, {}
    bv = -1.0
    for r in recs:
        if "val" in r and r["val"]["acc"] > bv:
            bv, bve, vpk = r["val"]["acc"], r["epoch"], r["val"].get("per_kind", {})
        if "holdout" in r and r["holdout"]["acc"] > bh:
            bh, bhe, hpk = r["holdout"]["acc"], r["epoch"], r["holdout"].get("per_kind", {})
    print(f"{v}\t{n_train}\t{bh:.4f}\t{bhe}\t{json.dumps(hpk, sort_keys=True)}\t{bve}\t{json.dumps(vpk, sort_keys=True)}\t{ts:.0f}")
EOF
echo done > "$OUT/DONE"
