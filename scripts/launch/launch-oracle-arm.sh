#!/usr/bin/env bash
# Hidden-information bound, part (a) (PLAN §0a "Hidden-information bound", user 2026-10-09: "Can you write step 2 into a
# plan and implement it and start now?"; review docs/reviews/2026-10-09-history-arch-opus.md Stage 1a): hppo's search
# (k4 s4 h1, value_ppo3, conf-skip 0.9) with the TRUE hidden state (--honest 0 = oracle rollouts) on P1's 744 deals vs
# ygo-agent. Paired: `paired_gate.py runs/p1-20261009-0922/hppo runs/<N>/oppo`. Gate oracle − honest ≥ +3 pp, p < 0.1.
# --claim 1: dynamic game queue (no straggler tail).
cd "$(dirname "$0")/../.."
for f in data/cache/base_r3/best.pt data/cache/value_ppo3/best.pt data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=oracle-$(date +%Y%m%d-%H%M); echo $N
B="--opp ygoagent --honest 0 --value data/cache/value_ppo3/best.pt --horizon 1 --k 4 --samples 4 --claim 1"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt SM_ARGS='--ga-seeds 3 --conf-skip 0.9' VARIANTS='oppo|$B' PROBE_ARGS='$B' VAR_CAP=180m PROBE_NPROC=30 bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-3.5}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 20 --extra-query "cpu_ram>=64" \
  --data decks data/cards/emb data/cache/base_r3 data/cache/value_ppo3 data/models > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
