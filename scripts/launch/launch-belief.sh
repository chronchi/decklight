#!/usr/bin/env bash
# Hidden-information bound, part (b) (PLAN §0a "Hidden-information bound", user 2026-10-09: "Can you write step 2 into a
# plan and implement it and start now?"; review docs/reviews/2026-10-09-history-arch-opus.md Stage 1b): a belief head
# on the base_r3 trunk predicts which opponent cards are hidden (hand / face-down), vs the hypergeometric baseline the
# determinizer uses. Variants: h128 (128 salient history entries), v1 (today's 32-entry window), nohist (no history
# tokens: isolates history's share), frozen128 (only new parameters train: what base_r3's features already hold).
# Gate: BCE ratio <= 0.9 on held-out games (val), <= 0.95 on held-out decks.
# Shards are built on the laptop first (engine only):
#   taskset -c 0-3 uv run python scripts/build_belief_shards.py --records data/demos/{teacher,sw-t1,search,window,\
#     dagger,sw-v,sw-d3,sw-v2}/'*.jsonl' --out data/cache/belief_shards --stride 4 --hist 128 --jobs 4
cd "$(dirname "$0")/../.."
for f in data/cache/base_r3/best.pt data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
[ -n "$(ls data/cache/belief_shards/*.npz 2>/dev/null | head -1)" ] || { echo "no belief shards"; exit 3; }
N=belief-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=data/cache/base_r3/best.pt EPOCHS=${EPOCHS:-3} PAR=${PAR:-2} bash scripts/vast_belief.sh $N" \
  --hours "${HOURS:-2}" --max-dph "${MAX_DPH:-0.30}" --min-cpus 8 --min-gpu-ram 20 \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64}" \
  --data data/cards/emb data/cache/base_r3 data/cache/belief_shards > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
