#!/usr/bin/env bash
# RL 5 continuation (user 2026-10-08: "Can we increase the time? So we have at least 400 iterations?"): the running
# job's time cap was fixed at launch (RL_HOURS 4.5, ~it 210), so a second box resumes it with train_rl --resume to
# --iters 400. Usage: scripts/launch/launch-rl5-resume.sh <run name of the segment to continue>
# Prepares data/cache/rl5_resume/ from runs/<run>/rl: resume.pt (a hardlink of last.pt made on the box, so the
# job's `rm last.pt` keeps it) → last.pt, curriculum.json, and the iter-NNNN.pt snapshot checkpoints. Same RL args as
# launch-rl5-league.sh (curriculum min_n/eps and --dags-starts must match: train_rl fails loud otherwise), except
# --iters 400 and --seed 1 (fresh game streams). KEEP_LAST=1 keeps this segment's last.pt for a further resume.
set -euo pipefail
cd "$(dirname "$0")/../.."
PREV=${1:?usage: launch-rl5-resume.sh <previous run name>}
SRC=runs/$PREV/rl; R=data/cache/rl5_resume
CK=$SRC/last.pt; [ -s "$CK" ] || CK=$SRC/resume.pt
for f in "$CK" "$SRC/curriculum.json" data/cache/value_ppo3/best.pt data/cache/base_r2/best.pt \
  data/cache/dags/starts-train-1k.jsonl; do [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
rm -rf "$R"; mkdir -p "$R"
cp "$CK" "$R/last.pt"; cp "$SRC/curriculum.json" "$R/"
IT=$(taskset -c 0-3 uv run --no-sync python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['iter'])" "$R/last.pt")
for k in $(seq 40 40 "$IT"); do cp "$SRC/iter-$(printf %04d "$k").pt" "$R/" || { echo "missing snapshot $k"; exit 3; }; done
[ -f "$SRC/iter-$(printf %04d "$IT").pt" ] && cp "$SRC/iter-$(printf %04d "$IT").pt" "$R/"
echo "resume from $PREV at iteration $IT: $(ls "$R" | tr '\n' ' ')"
N=rl5resume-$(date +%Y%m%d-%H%M); echo $N
PTA='--decks 8 --hands 3 --seeds 8 --narrate 0 --samples 4 --ygoagent --ga-seeds 3 --newcards decks/newcards'
RLA="--lr 6e-5 --kl-coef 0.05 --league 1 --p-self 0.3 --ga-episode-share 0.5 --ga-ygo-share 0.5 --league-fixed ft-swv=data/cache/base_r2/best.pt --league-fixed-share 0.10 --league-greedy-share 0.05 --snap-every 40 --league-size 6 --priv-critic 1 --critic-init data/cache/value_ppo3/best.pt --dags-starts data/cache/dags/starts-train-1k.jsonl --dags-p 0.3 --curric 1 --ga-workers 16 --iters 400 --seed 1 --resume $R"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "INIT=data/cache/base_r3/best.pt KEEP_LAST=1 RL_HOURS=${RL_HOURS:-4.6} EVAL_EVERY=40 CARD_K=2 CARD_BY=vs_ygoagent CARD_PT_ARGS='$PTA' RL_ARGS='$RLA' bash scripts/vast_rl.sh $N" \
  --hours "${HOURS:-6.8}" --max-dph 0.35 --min-cpus 32 --min-gpu-ram 20 --extra-query "cpu_ram>=48" \
  --data decks data/demos data/cards/emb data/models data/cache/base_r3 data/cache/base_r2 data/cache/value_ppo3 data/cache/dags "$R" \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
