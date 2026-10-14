#!/usr/bin/env bash
# Retrain stage B1 (docs/retrain/plan-v2.md §Stage B1, plan-v3; go: user 2026-10-11 "yeah, go ahead"): two B-v1s from
# scratch (clean, clean + hw) on the stage D clean mix + stage M turns, then the raw card on 5,952 fresh deals vs
# ygo-agent with ft-swv / ft-teach-h / base_r3 references, and holdout/new-card head-to-heads vs ft-swv.
# Needs: the stage M sweep and its play-out anchor (ANCHOR_DROPS = jsonl {"line": id} of the turn lines it dropped).
# B1 runs the clean arm only (user 2026-10-12: option d, "I'll go with d, makes more sense"; NOTEBOOK 07:45).
# Pass-weight test (NOTEBOOK 2026-10-13 19:00): PREFIX=passw EXTRA_FRAC="teacher_pass=2 dagger_pass=2" CARD=0 MS_EVAL=1.
# Env: VARIANTS (default "clean"; "clean hw" = the plan-v2 two-arm design), ANCHOR_DROPS, HW_FRAC (default 1), EPOCHS (6), HOURS (5), MAX_DPH (0.32), SMOKE=1 (tiny, ~20 min).
cd "$(dirname "$0")/../.."
for f in data/cache/base_r3/best.pt data/cache/base_r2/best.pt data/cache/base_teach_h/best.pt data/models/0546_26550M.tflite \
         data/cards/emb/qwen3-0.6b-d512.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
if [ "${NO_TURNS:-0}" != 1 ]; then
  [ -s "${ANCHOR_DROPS:-}" ] || { echo "set ANCHOR_DROPS (turn-anchor drops jsonl) or NO_TURNS=1"; exit 3; }
  ls data/demos/turns/*/*.jsonl > /dev/null 2>&1 || { echo "no stage M turns in data/demos/turns"; exit 3; }
  # only anchored turn lines train: lines verified after the anchor ran (stage M split mode keeps adding them,
  # NOTEBOOK 2026-10-12 21:00) are excluded with the anchor's drops
  EX="$(dirname "$ANCHOR_DROPS")/b1_excl.jsonl"
  uv run --no-sync python - "$ANCHOR_DROPS" "$EX" <<'PY' || exit 3
import glob, json, sys
from pathlib import Path
drops, out = Path(sys.argv[1]), Path(sys.argv[2])
seen = {json.loads(x)["line_id"] for f in glob.glob(str(drops.parent / "anchor" / "p*.jsonl")) for x in open(f) if x.strip()}
if not seen:
    sys.exit(f"no anchor records next to {drops}")
ids = [json.loads(x)["line"] for x in drops.read_text().splitlines() if x.strip()]
turns = [json.loads(x)["line_id"] for f in glob.glob("data/demos/turns/*/*.jsonl") for x in open(f) if x.strip()]
late = [t for t in turns if t not in seen]
out.write_text("".join(json.dumps({"line": i}) + "\n" for i in sorted(set(ids) | set(late))))
print(f"turn lines: {len(turns)}; anchored {len(seen)}, anchor drops {len(ids)}, not anchored {len(late)} -> {out}")
PY
  ANCHOR_DROPS=$EX
fi
N=${PREFIX:-stage-b1}-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
D=(data/demos/teacher data/demos/dagger data/demos/gold data/demos/verified data/demos/expanded data/demos/claude
   data/demos/planned data/demos/planned-s1 data/demos/planned-s2 data/demos/planned-s3 data/demos/planned-s4
   data/demos/guides-strict-d data/demos/sw-hw data/cache/audit data/cards/emb decks data/models
   data/cache/base_r3 data/cache/base_r2 data/cache/base_teach_h)
[ "${NO_TURNS:-0}" != 1 ] && D+=(data/demos/turns "$(dirname "$ANCHOR_DROPS")")
[ "${MS_EVAL:-0}" = 1 ] && D+=(data/cache/bv1)  # pass-weight test: B-v1 (P's init) is the eval baseline
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "EXTRA_FRAC='${EXTRA_FRAC:-}' CARD=${CARD:-1} MS_EVAL=${MS_EVAL:-0} VARIANTS='${VARIANTS:-clean}' ANCHOR_DROPS=${ANCHOR_DROPS:-} NO_TURNS=${NO_TURNS:-0} HW_FRAC=${HW_FRAC:-1} EPOCHS=${EPOCHS:-6} SMOKE=${SMOKE:-0} GA_SEEDS=${GA_SEEDS:-24} bash scripts/vast_stage_b1.sh $N" \
  --hours "${HOURS:-5}" --max-dph "${MAX_DPH:-0.32}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-20}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64}" \
  --data "${D[@]}" > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
