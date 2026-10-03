#!/usr/bin/env bash
# One GPU session (PLAN Phase 4): card embeddings (GPU) || data backlog (CPU), then BC shards (CPU), then BC training (GPU).
# Usage: bash scripts/vast_bc.sh <name> [epochs]. Everything to keep goes to runs/<name>/ (pulled back by the launcher).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; EPOCHS="${2:-30}"; OUT="runs/$NAME"; mkdir -p "$OUT"
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
if [ "$Q" = "max" ]; then J=$(nproc); else J=$(( Q / P )); fi
[ "$J" -lt 2 ] && J=2
export HF_HOME="$PWD/data/cache/hf"
uv sync --group dev --group ml -q
uv pip install -q --reinstall torch --index-url https://download.pytorch.org/whl/cu128 > "$OUT/torch.txt" 2>&1 || true
{ nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; echo "jobs $J";
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; } > "$OUT/machine.txt" 2>&1
T0=$SECONDS
# 1. GPU: embeddings, while the CPU cores expand the newest verified lines.
uv run --no-sync python scripts/embed_cards.py --dim 512 --batch 256 > "$OUT/embed.txt" 2>&1 &
EMB=$!
H=$(( J > 4 ? J - 2 : 2 ))
for spec in "data/demos/claude/verified/kewl-tune.jsonl data/demos/claude/lines data/demos/claude/expanded-kewl-tune" \
            "data/demos/planned/verified/mitsurugi.jsonl data/demos/planned/lines data/demos/planned/expanded-mitsurugi" \
            "data/demos/planned/verified/branded.jsonl data/demos/planned/lines data/demos/planned/expanded-branded"; do
  set -- $spec
  [ -s "$1" ] && uv run --no-sync python scripts/expand_lines.py --verified "$1" --lines-dir "$2" --out "$3" --variants 8 --jobs "$H" \
    >> "$OUT/expand.txt" 2>&1
done
wait $EMB
echo "embed+expand $((SECONDS - T0)) s" > "$OUT/timing.txt"
mkdir -p "$OUT/emb" "$OUT/expanded"; cp data/cards/emb/*.npz "$OUT/emb/"
for d in data/demos/claude/expanded-kewl-tune data/demos/planned/expanded-mitsurugi data/demos/planned/expanded-branded; do
  [ -d "$d" ] && cp -r "$d" "$OUT/expanded/$(echo "$d" | tr / _)"
done
# 2. CPU: shards from every record source.
uv run --no-sync python scripts/build_bc_shards.py --out data/cache/bc_shards --jobs "$J" > "$OUT/shards.txt" 2>&1
echo "shards $((SECONDS - T0)) s" >> "$OUT/timing.txt"
# 3. GPU: BC training (holdout deck: kewl-tune, never trained on).
uv run --no-sync python scripts/train_bc.py --shards data/cache/bc_shards --emb data/cards/emb/qwen3-0.6b-d512.npz \
  --out "$OUT/bc" --epochs "$EPOCHS" --batch 512 --lr 5e-4 --holdout kewl-tune > "$OUT/train.txt" 2>&1
echo "train $((SECONDS - T0)) s" >> "$OUT/timing.txt"
echo done > "$OUT/DONE"
