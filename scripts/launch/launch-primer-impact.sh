#!/usr/bin/env bash
# Impact-ranked opponent primers (primer-impact): score the hppo bot-vs-bot games (31 gauntlet decks) and the
# primer games (189 lobby decks) with the product value net, one build_primers.py --score run on a many-core
# GPU box. Env: HOURS (1), MAX_DPH (0.30), MIN_CPUS (32), MIN_RAM (64).
# Results (pulled to the laptop): runs/<name>/primers.json + summaries.json.gz. Do NOT launch from a brief.
cd "$(dirname "$0")/../.."
. scripts/product.sh
G1=runs/value-p2-20261015-2044; G2=runs/primer-games-20261018-0055
for d in $G1/hppo-p $G1/hppo-p2 $G2/pool $G2/newcards; do ls $d/games-*.jsonl > /dev/null 2>&1 || { echo "no games in $d"; exit 3; }; done
[ -f "$PRODUCT_VALUE" ] || { echo "missing $PRODUCT_VALUE"; exit 3; }
N=primer-impact-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "bash scripts/vast_primer_impact.sh $N $G1/hppo-p $G1/hppo-p2 $G2/pool $G2/newcards" \
  --hours "${HOURS:-1}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" \
  --extra-query "cpu_ram>=${MIN_RAM:-64}" \
  --data decks data/cards/emb data/cache/P "$PRODUCT_VALUE_DIR" $G1/hppo-p $G1/hppo-p2 $G2/pool $G2/newcards \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
