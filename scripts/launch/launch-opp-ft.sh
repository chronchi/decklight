#!/usr/bin/env bash
# Fine-tune P on opponent-turn window labels + the pre-registered gates (scripts/vast_opp_ft.sh, NOTEBOOK 2026-11-04 16:00).
# Env: WIN (runs/<opp-windows job>/window, required), EVAL_STARTS (starts-v2.eval.jsonl), ANCHOR_GAMES (640), WREP (8),
#      CAND_CKPT (gates only on this checkpoint; its directory is shipped), KLI (0.3), EPOCHS (2), LR (1e-4), SMOKE=1, HOURS (5), MAX_DPH (0.45), MIN_CPUS (24), MIN_GPU_RAM (16).
cd "$(dirname "$0")/../.."
EVAL_STARTS=${EVAL_STARTS:-data/cache/line_starts/starts-v2.eval.jsonl}
for f in "${WIN:?set WIN}" "$EVAL_STARTS" data/cache/forge_r1/best.pt data/cache/value_p2/best.pt; do
  [ -e "$f" ] || { echo "missing $f"; exit 3; }
done
ls "$WIN"/*.jsonl > /dev/null || { echo "no window records in $WIN"; exit 3; }
python3 - "$WIN" "$EVAL_STARTS" <<'PY' || exit 3
import glob, json, sys
held = {json.loads(x)["line_id"] for x in open(sys.argv[2]) if x.strip()}
used = {json.loads(x)["start_line_id"] for f in glob.glob(sys.argv[1] + "/*.jsonl") for x in open(f) if x.strip()}
if not used or used & held:
    sys.exit(f"window records: {len(used)} start lines, {len(used & held)} held out, e.g. {sorted(used & held)[:3]}")
PY
N=opp-ft-$(date +%Y%m%d-%H%M)${CAND_CKPT:+-gates}${SMOKE:+-smoke}; echo $N
rm -rf data/cache/oppft_win && mkdir -p data/cache/oppft_win/window && cp "$WIN"/*.jsonl data/cache/oppft_win/window/
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "CAND_CKPT=${CAND_CKPT:-} SMOKE=${SMOKE:-0} WIN_DIR=data/cache/oppft_win/window EVAL_STARTS=$EVAL_STARTS ANCHOR_GAMES=${ANCHOR_GAMES:-640} WREP=${WREP:-8} KLI=${KLI:-0.3} EPOCHS=${EPOCHS:-2} LR=${LR:-1e-4} bash scripts/vast_opp_ft.sh $N" \
  --hours "${HOURS:-5}" --max-dph "${MAX_DPH:-0.45}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-48} gpu_name!=CMP_170HX" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/forge_r1 data/cache/value_p2 data/cache/line_starts data/cache/oppft_win ${CAND_CKPT:+$(dirname "$CAND_CKPT")} \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
