#!/usr/bin/env bash
# Opponent-turn plan step 2 (docs/plans/2026-11-03-opp-turn-interaction.md; user: "agreed with the 60/20/20", 2026-11-03
# "You can work on the whole plan during the night"): PPO from the product P (forge_r1) with combo-line opponents.
# Stage P's job (vast_stage_p.sh: PPO recipe of P3, milestone, ygo-agent card, principles) with INIT = REF = P, plus:
#   --line-starts (60% of episodes, --dags-p 0.6): the opponent replays an engine-verified combo line until our bot's
#     interaction changes it, then P continues (src/decklight/rl/lines.py);
#   league on the other 40%: half gauntlet episodes vs ygo-agent (--ga-episode-share 0.5 --ga-ygo-share 1.0), half
#     P / snapshots (--p-self 0) -> 60/20/20;
#   the combo-line card on held-out lines (LINE_EVAL_STARTS, scripts/audit/line_eval.py).
# Env: STARTS (data/cache/line_starts/starts-v1.jsonl; its .eval.jsonl is the held-out set), SMOKE=1, HOURS (8),
#      MAX_DPH (0.40), ITERS (560), MS_IT (160), RL_HOURS (5.5), RL_EXTRA (more train_rl flags), GA_SEEDS (24).
cd "$(dirname "$0")/../.."
STARTS=${STARTS:-data/cache/line_starts/starts-v1.jsonl}; EV=${STARTS%.jsonl}.eval.jsonl
for f in data/cache/forge_r1/best.pt data/cache/forge_r1/metrics.jsonl data/cache/value_p2/best.pt "$STARTS" "$EV" \
         data/models/0546_26550M.tflite data/cards/emb/qwen3-0.6b-d512.npz data/cards/lua_desc.npz; do
  [ -s "$f" ] || { echo "missing $f"; exit 3; }
done
LX="--line-starts $STARTS --dags-p 0.6 --league 1 --p-self 0 --ga-episode-share 0.5 --ga-ygo-share 1.0 --league-size 6 --snap-every 40 --max-upload-gb 1"
N=opp-ppo-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "SMOKE=${SMOKE:-0} SMOKE_ITERS=${SMOKE_ITERS:-6} INIT=data/cache/forge_r1/best.pt REF=data/cache/forge_r1/best.pt DEAL0=15000 ITERS=${ITERS:-560} MS_IT=${MS_IT:-160} MS_MIN=${MS_MIN:-0.50} RL_HOURS=${RL_HOURS:-5.5} RL_EXTRA='$LX ${RL_EXTRA:-}' LINE_EVAL_STARTS=$EV LINE_REPEATS=${LINE_REPEATS:-4} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GA_SEEDS=${GA_SEEDS:-24} bash scripts/vast_stage_p.sh $N" \
  --hours "${HOURS:-8}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-24}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-48} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 data/cache/line_starts \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
