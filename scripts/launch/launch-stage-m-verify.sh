#!/usr/bin/env bash
# Retrain stage M, split mode (scripts/vast_stage_m.sh; user 2026-10-12 "Agreed with 1 and 2"): verify on a many-core box
# from the laptop's LLM cache. Before round 1, on the laptop (LLM only, no engine search):
#   plan_turns.py --games 'data/demos/sw-v/*.jsonl' --per-bucket 4 --deck-groups data/cache/stage_m/groups-new.txt --llm-only
#   plan_turns.py ... --deck-groups data/cache/stage_m/groups-old.txt --llm-only --ids-out data/cache/stage_m/ids-old.txt
# After a round: merge runs/<name>/turns/*/*.jsonl into data/demos/turns/<bucket>/, answer runs/<name>/pending.jsonl with
#   plan_turns.py --games 'data/demos/sw-v/*.jsonl' --llm-only --prompts runs/<name>/pending.jsonl
# and launch round 2 with the same env (verified states are skipped; repairs now have a cached answer).
# Ships the LLM cache (responses only; never the OpenCode key). CPU work: cheap GPU listings with many cores.
# Env: NEW_GROUPS (data/cache/stage_m/groups-new.txt), OLD_GROUPS + IDS (optional), BUDGET, CAP, HOURS, MAX_DPH, MIN_CPUS.
cd "$(dirname "$0")/../.."
NEW_GROUPS=${NEW_GROUPS-data/cache/stage_m/groups-new.txt}
for f in $NEW_GROUPS ${OLD_GROUPS:-} ${IDS:-}; do [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
N=stage-m-$(date +%Y%m%d-%H%M); echo $N
# data/strategy: the per-deck notes are part of every prompt (without them no prompt matches the cache: stage-m-20261012-2250)
D=(decks data/demos/sw-v data/demos/turns data/cache/llm data/cache/stage_m data/strategy)
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "NEW_GROUPS='$NEW_GROUPS' OLD_GROUPS='${OLD_GROUPS:-}' IDS='${IDS:-}' BUDGET=${BUDGET:-8000} CAP=${CAP:-150m} bash scripts/vast_stage_m.sh $N" \
  --hours "${HOURS:-3}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-64}" --min-gpu-ram 4 \
  --data "${D[@]}" > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
