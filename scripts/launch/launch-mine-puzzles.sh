#!/usr/bin/env bash
# Full puzzle mining over the hppo bot-vs-bot games (31 gauntlet decks) and the primer games (189 lobby decks), one
# process per games file on a many-core box (CPU verification, fast-mode bot). Env: HOURS (1), MAX_DPH (0.40),
# MIN_CPUS (32), PER_FILE (4), FILE_CAP (40m). Merge on the laptop: scripts/merge_puzzles.py runs/<name>/shards.
cd "$(dirname "$0")/../.."
. scripts/product.sh
G1=runs/value-p2-20261015-2044; G2=runs/primer-games-20261018-0055
for d in $G1/hppo-p $G1/hppo-p2 $G2/pool $G2/newcards; do ls $d/games-*.jsonl > /dev/null 2>&1 || { echo "no games in $d"; exit 3; }; done
N=mine-puzzles-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "PER_FILE=${PER_FILE:-4} FILE_CAP=${FILE_CAP:-40m} bash scripts/vast_mine_puzzles.sh $N $G1/hppo-p $G1/hppo-p2 $G2/pool $G2/newcards" \
  --hours "${HOURS:-1}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" \
  --extra-query "cpu_ram>=${MIN_RAM:-64}" \
  --data decks data/cards/emb data/cache/P "$PRODUCT_VALUE_DIR" $G1/hppo-p $G1/hppo-p2 $G2/pool $G2/newcards \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
