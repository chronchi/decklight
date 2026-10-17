#!/usr/bin/env bash
# Custom-card games (scripts/vast_custom_games.sh): user 2026-10-17, a friend's Wiccanthrope custom cards
# (github.com/zedja123/ZedjaCustomCards, cloned to data/cache/custom/zedja, audited: no os/io/require/load).
# 3 opponents x 2 seats x SEEDS deal seeds, one process per game. Env: HOURS (2.5), MAX_DPH (0.45), OPPS, SEEDS (2).
cd "$(dirname "$0")/../.."
. scripts/product.sh
for f in "$PRODUCT_CKPT" "$PRODUCT_VALUE" data/cards/emb/qwen3-0.6b-d512.npz data/cards/custom/lua_desc-zedja.npz \
         decks/custom/wiccanthrope.ydk data/cache/custom/zedja/Wiccanthrope.cdb; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=custom-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "EXTRA=data/cache/custom/zedja CUSTOM=../custom/wiccanthrope OPPS=${OPPS:-snake-eye,mitsurugi-meta2,tearlaments} SEEDS=${SEEDS:-2} bash scripts/vast_custom_games.sh $N" \
  --hours "${HOURS:-2.5}" --max-dph "${MAX_DPH:-0.45}" --min-cpus "${MIN_CPUS:-16}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --extra-query "compute_cap>=750 cpu_ram>=${MIN_RAM:-32}" \
  --data decks data/cards data/models data/cache/P "$PRODUCT_VALUE_DIR" data/cache/custom/zedja \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
