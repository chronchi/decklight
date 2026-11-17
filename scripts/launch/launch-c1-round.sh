#!/usr/bin/env bash
# Cycle-1 verify / distill / card (docs/lab/prereg/2026-11-10-master-cycle1.md, amendment 3): the packs trained by
# scripts/launch/launch-c1-train.sh, merged into data/forge/c1/packs, are verified against X, the accepted ones are
# distilled into X2 (observable history, OBS_HIST=1) and the release card is played (G1-G3; G4/G5 are separate jobs).
# Usage: MAX_DPH=0.30 bash scripts/launch/launch-c1-round.sh
cd "$(dirname "$0")/../.."
X=runs/window-ppo-20261108-0940/train/export-X/policy
P=${PACKS:-data/forge/c1/packs}
for f in $X/best.pt $X/metrics.jsonl data/cache/x_window/best.pt data/cache/value_p2/best.pt data/cache/P/best.pt \
         data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz data/cards/lua_desc.npz \
         data/models/0546_26550M.tflite decks/forge/c1/split.json decks/forge/f2/pairs/reserve.pairs; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
n=0; for d in "$P"/*/; do [ -f "$d/pack.json" ] || { echo "missing $d/pack.json"; exit 3; }; n=$((n + 1)); done
[ "$n" -ge 1 ] || { echo "no packs in $P"; exit 3; }
OLD='blitzclique elfnote fiendsmith-yummy ignister-maliss kewl-tune lunalight magistus-invoked-fairy-tail mitsurugi'
OLD="$OLD power-patron-artmage trains branded-light-and-darkness-ritual radiant-typhoon sacred-beasts atlantean-mermail"
N=c1-round-$(date +%Y%m%d-%H%M); echo "$N ($n packs)"
ENVS="ARCHS=none REL=$X/best.pt REL_NAME=X NEXT_NAME=X2 SPLIT=decks/forge/c1/split.json PACKS_IN=$P OBS_HIST=1"
ENVS="$ENVS OLD_ARCHS='$OLD' OLD_PAIRS=decks/forge/c1/old-pairs OLD_GAMES=1536"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "$ENVS bash scripts/vast_forge_round.sh $N" \
  --hours "${HOURS:-12}" --max-dph "${MAX_DPH:?}" --min-cpus "${MIN_CPUS:-48}" --min-gpu-ram "${MIN_GPU_RAM:-22}" \
  --reserve "${RESERVE:-0.10}" --max-inet-cost "${MAX_INET:-0.005}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64} ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/x_window data/cache/value_p2 $X $P \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
