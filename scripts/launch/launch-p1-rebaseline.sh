#!/usr/bin/env bash
# P1 (PLAN §0a "Plan after step 3", user 2026-10-09: "Ok I approve except the scale up"): E2 re-run on the same 744
# deals with the card-faithful loop guard (NOTEBOOK 2026-10-09 08:40: E2's shared guard forbade ygo-agent's own pick at
# repeated chain windows, 596/744 games differed from the card). Raw arm first: it must reproduce the card's base_r3
# games move for move (EXPECT, scripts/audit/harness_check.py, >= 99%) or the search arm is skipped. Then hppo = E2's
# search (k4 s4 h1, value_ppo3, honest, det-tries 32, skip), now logging per-sample rollout scores. hppo is the new
# reference every P2 arm pairs with: `scripts/audit/paired_gate.py runs/<N>/raw runs/<N>/hppo`.
# Value lever (user 2026-10-09: "I find it interesting that a different value function could help more here"), run in
# the same box on the same deals: hpriv = hppo with rl6's privileged critic as leaf value (--value-priv 1; init
# value_ppo3, 218 PPO iters with hidden cards revealed). Honest rollouts run in sampled worlds, so the critic reads
# that world's cards. Paired: `paired_gate.py runs/<N>/hppo runs/<N>/hpriv`.
cd "$(dirname "$0")/../.."
REF=data/cache/harness/card-base_r3-744.jsonl
for f in data/cache/base_r3/best.pt data/cache/value_ppo3/best.pt data/cache/critic_rl6/critic.pt data/models/0546_26550M.tflite $REF; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=p1-$(date +%Y%m%d-%H%M); echo $N
H="--opp ygoagent --honest 1 --det-tries 32 --det-fallback skip --value data/cache/value_ppo3/best.pt --horizon 1"
P="--opp ygoagent --honest 1 --det-tries 32 --det-fallback skip --value data/cache/critic_rl6/critic.pt --value-priv 1 --horizon 1"
V="raw|--opp ygoagent --raw 1;hppo|$H;hpriv|$P"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt SM_ARGS='--ga-seeds 3 --k 4 --samples 4 --conf-skip 0.9' VARIANTS='$V' PROBE_ARGS='$H' EXPECT='raw:$REF' VAR_CAP=150m PROBE_NPROC=30 bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-4.5}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 20 --extra-query "cpu_ram>=64" \
  --data decks data/cards/emb data/cache/base_r3 data/cache/value_ppo3 data/cache/critic_rl6 data/models data/cache/harness > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
