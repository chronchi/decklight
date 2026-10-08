#!/usr/bin/env bash
# Strategy review E2 (PLAN §0a step 2): does test-time search beat ygo-agent? base_r3 + honest search (value_ppo3,
# --det-fallback skip: no oracle fallback) vs ygo-agent, and raw base_r3 (--raw 1, same code path) vs ygo-agent, on
# the same 744 deals (31 gauntlet decks x next 4 x 3 seeds x 2 seats). Paired with
# `scripts/audit/paired_gate.py runs/<N>/raw runs/<N>/hppo`. Search variant first: the sizing probe runs on it
# (PROBE_NPROC=30: ~25 games, so the probe outlives its 150 s window; at 1000 it got 1 game and ended in 3 s).
# Reference: raw base_r3 vs ygo-agent 42.3% (248 deals, playtest). S0b hppo vs raw base_r3: 0.688.
cd "$(dirname "$0")/../.."
for f in data/cache/base_r3/best.pt data/cache/value_ppo3/best.pt data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=e2-$(date +%Y%m%d-%H%M); echo $N
V="hppo|--opp ygoagent --honest 1 --det-tries 32 --det-fallback skip --value data/cache/value_ppo3/best.pt --horizon 1;raw|--opp ygoagent --raw 1"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt SM_ARGS='--ga-seeds 3 --k 4 --samples 4 --conf-skip 0.9' VARIANTS='$V' VAR_CAP=150m PROBE_NPROC=30 bash scripts/vast_search_match.sh $N" \
  --hours 4 --max-dph 0.40 --min-cpus 32 --min-gpu-ram 20 --extra-query "cpu_ram>=64" \
  --data decks data/cards/emb data/cache/base_r3 data/cache/value_ppo3 data/models > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
