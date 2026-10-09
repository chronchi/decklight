#!/usr/bin/env bash
# P2 (PLAN §0a "Plan after step 3", user 2026-10-09: "Ok I approve except the scale up"): search variants on P1's 744
# deals vs ygo-agent, each paired with P1's search arm (`paired_gate.py runs/p1-20261009-0922/<ref> runs/<N>/<arm>`).
# P1's per-sample audit (NOTEBOOK 17:05): override margin median 0.068, only 15.6% of overrides >= 2 SE.
# - t1:   own turn 1 searched, S going first only (going-second games are identical to the reference's);
# - m05 / m10: override the policy only if the search's best beats its move by >= 0.05 / 0.1 (57% / 40% of overrides);
# - k2s2: 2 candidates x 2 samples (budget calibration: ~1/4 of the rollouts).
# VALUE: leaf value args of P1's winner (hppo: value_ppo3; hpriv: privileged critic). --claim 1: dynamic game queue
# (P1's static shards idled 17 of 18 procs for ~50 min behind one slow shard).
cd "$(dirname "$0")/../.."
VALUE=${VALUE:?set VALUE, e.g. "--value data/cache/value_ppo3/best.pt"}
for f in data/cache/base_r3/best.pt data/cache/value_ppo3/best.pt data/cache/critic_rl6/critic.pt data/models/0546_26550M.tflite; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
N=p2-$(date +%Y%m%d-%H%M); echo $N
B="--opp ygoagent --honest 1 --det-tries 32 --det-fallback skip $VALUE --horizon 1 --claim 1"
V="t1|$B --k 4 --samples 4 --own-turn1 1 --s-seat-only 0;m05|$B --k 4 --samples 4 --margin 0.05;m10|$B --k 4 --samples 4 --margin 0.1;k2s2|$B --k 2 --samples 2"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CKPT=data/cache/base_r3/best.pt SM_ARGS='--ga-seeds 3 --conf-skip 0.9' VARIANTS='$V' PROBE_ARGS='$B --k 4 --samples 4' VAR_CAP=150m PROBE_NPROC=30 bash scripts/vast_search_match.sh $N" \
  --hours "${HOURS:-6}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram 20 --extra-query "cpu_ram>=64" \
  --data decks data/cards/emb data/cache/base_r3 data/cache/value_ppo3 data/cache/critic_rl6 data/models > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
