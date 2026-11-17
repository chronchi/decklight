#!/usr/bin/env bash
# Value v2 training + gate V1 (docs/lab/prereg/2026-11-11-value-v2.md; scripts/vast_value_v2.sh). Stages data/cache/value_v2:
# Y (product policy), Yv (Y's league critic, the warm start), value_p2 (the comparison), the 25 c1 + 10 f3 specialist
# packs (sky-striker dropped in both, as in the c1 league), the V1 held-out games (150 per c1-card N row of Y, not
# trained on) and their decks. Env: HOURS (4.5), MAX_DPH (0.26), MIN_CPUS (32), MIN_GPU_RAM (16), MIN_RAM (60),
# MAX_INET (0.015), V2_FROM (an export dir: V1 only on its value net). The product dirs ship too: config/bots/*.toml validate them (tsoj's dl bot). Plus the job's env (VITERS, ...).
cd "$(dirname "$0")/../.."
. scripts/product.sh
G=data/cache/value_v2; YE=runs/c1-league-20261111-0821/train/export-Y; CARD=runs/c1-league-20261111-1057/card
rm -rf $G; mkdir -p $G/Y $G/Yv $G/value_p2 $G/packs $G/v1games $G/decks
cp $PRODUCT_CKPT_DIR/{best.pt,metrics.jsonl} $G/Y/ && cp $YE/value/{best.pt,metrics.jsonl} $G/Yv/ \
  && cp $PRODUCT_VALUE_DIR/{best.pt,metrics.jsonl} $G/value_p2/ && cp data/cache/c1_league/card-decks/*.ydk $G/decks/ \
  || { echo "staging failed"; exit 3; }
cmp -s $G/Y/best.pt $YE/policy/best.pt || { echo "product policy is not Y"; exit 3; }
for d in data/forge/c1/packs/*/ data/forge/f3/packs/*/; do
  a=$(basename "$d"); [ "$a" = sky-striker ] && continue
  [ -d $G/packs/$a ] && { echo "pack name clash: $a"; exit 3; }
  mkdir -p $G/packs/$a/policy $G/packs/$a/decks
  cp "$d"pack.json $G/packs/$a/ && cp "$d"policy/metrics.jsonl $(ls "$d"policy/weights.safetensors "$d"policy/best.pt 2>/dev/null | head -1) $G/packs/$a/policy/ \
    && cp -r "$d"decks/train $G/packs/$a/decks/ || { echo "staging pack $a failed"; exit 3; }
done
python3 scripts/tsoj/deck_opps.py --packs $G/packs --want 35 --out $G/opps || exit 3
for d in $CARD/N-*/; do  # 150 games per row, deterministic
  cat "$d"games-*.jsonl | head -150 > $G/v1games/games-$(basename "$d").jsonl
done
if [ -n "${V2_FROM:-}" ]; then  # V1 only on an exported v2 (its value/best.pt + metrics.jsonl)
  mkdir -p $G/v2 && cp "$V2_FROM"/value/{best.pt,metrics.jsonl} $G/v2/ || { echo "staging v2 failed"; exit 3; }
fi
du -sh $G; echo "V1 games: $(cat $G/v1games/*.jsonl | wc -l) from $(ls $G/v1games | wc -l) rows"
N=value-v2-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "VITERS=${VITERS:-120} ${V2_FROM:+V2_IN=$G/v2/best.pt} bash scripts/vast_value_v2.sh $N" \
  --hours "${HOURS:-4.5}" --reserve "${RESERVE:-0.25}" --max-dph "${MAX_DPH:-0.26}" --min-cpus "${MIN_CPUS:-32}" \
  --min-gpu-ram "${MIN_GPU_RAM:-16}" --max-inet-cost "${MAX_INET:-0.015}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-60}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models $PRODUCT_CKPT_DIR $PRODUCT_VALUE_DIR $G \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
