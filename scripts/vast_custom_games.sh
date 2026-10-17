#!/usr/bin/env bash
# Custom-card games (user 2026-10-17: "I want to know if our agent at full capacity can play this deck. Get the
# embeddings appropriately then also run a few games against 3 different decks"). On one vast GPU box:
#   1. embed only the cards / effect strings missing from the stock npz files (YGO_EXTRA_CARDS dir), merged copies;
#   2. the product on both seats (search_match --both-search 1) with the merged files, CUSTOM vs each of OPPS,
#      both seats, SEEDS deal seeds each; one process per game.
# Env: EXTRA (custom card dir), CUSTOM (deck path relative to decks/pool), OPPS (pool decks), SEEDS. vast GPU only.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300 HF_HOME="$PWD/data/cache/hf"
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$OUT/machine.txt"; nproc >> "$OUT/machine.txt"
export YGO_EXTRA_CARDS="${EXTRA:?}"
EMB="$OUT/emb-custom.npz"; STR="$OUT/strings-custom.npz"
uv run --no-sync python scripts/embed_cards.py --dim 512 --batch 128 --only-missing \
  --base data/cards/emb/qwen3-0.6b-d512.npz --out "$EMB" > "$OUT/embed-cards.txt" 2>&1 || { alert "card embed failed"; exit 4; }
uv run --no-sync python scripts/embed_cards.py --dim 512 --batch 128 --only-missing --strings \
  --base data/cards/emb/strings-qwen3-0.6b-d512.npz --out "$STR" > "$OUT/embed-strings.txt" 2>&1 || { alert "string embed failed"; exit 4; }
# every custom card in the deck must now have a non-zero embedding
uv run --no-sync python - "$EMB" "$CUSTOM" <<'PY' >> "$OUT/PROGRESS" || { alert "custom cards missing from merged embeddings"; exit 5; }
import sys, numpy as np
from ygo_zero.decks.ydk import load_ydk
z = np.load(sys.argv[1]); idx = {int(c): i for i, c in enumerate(z["codes"])}
d = load_ydk(f"decks/pool/{sys.argv[2]}.ydk")
codes = sorted(set(d.main) | set(d.extra))
miss = [c for c in codes if c not in idx or not np.any(z["card"][idx[c]])]
print(f"embeddings: {len(z['codes'])} codes, deck {len(codes)} distinct, missing {miss}")
sys.exit(1 if miss else 0)
PY
export YGO_EMB="$EMB" YGO_STRINGS="$STR" YGO_LUA="data/cards/custom/lua_desc-zedja.npz"
. scripts/product.sh
IFS=',' read -ra OS <<< "${OPPS:?}"
PIDS=(); i=0
for o in "${OS[@]}"; do
  for pair in "$CUSTOM:$o" "$o:$CUSTOM"; do
    for s in $(seq 0 $(( ${SEEDS:-1} - 1 ))); do
      # shellcheck disable=SC2086
      OMP_NUM_THREADS=1 uv run --no-sync python scripts/search_match.py --ckpt "$PRODUCT_CKPT" --emb "$EMB" \
        --value "$PRODUCT_VALUE" --device cuda --opp self --both-search 1 $PRODUCT_SEARCH \
        --pairs "$pair" --seeds 1 --deal-seed0 $(( ${DEAL_SEED0:-20261017} + s )) \
        --s-seat-only 0 --width 1 --out "$OUT/g$i" > "$OUT/g$i.txt" 2>&1 &
      PIDS+=($!); echo "g$i $pair seed$s" >> "$OUT/games.txt"; i=$((i + 1))
    done
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || alert "game proc $pid exit $?"; done
for g in $(seq 0 $((i - 1))); do
  [ -s "$OUT/g$g/games-0.jsonl" ] || alert "game g$g wrote no record: $(tail -1 "$OUT/g$g.txt")"
done
echo "$(date +%T) custom games done: $(cat "$OUT"/g*/games-0.jsonl 2>/dev/null | wc -l) games" | tee -a "$OUT/PROGRESS"
