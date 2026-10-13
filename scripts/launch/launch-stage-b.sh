#!/usr/bin/env bash
# Retrain stage B (docs/retrain/plan-v2.md §Stage B, plan-v3): B-v3 (PolicyNetV3 + --hist) from scratch on B1's clean mix
# + every anchored stage M turn, first milestone (context-behaviour band) after epoch 1, then the card on B1's deals
# paired with B1's games, holdout/new-card head-to-heads vs ft-swv and the gate B principle audit.
# Needs: B1 (runs/stage-b1-20261013-0008: B-v1 clean weights + card games) and the stage M anchor over all turns
# (ANCHOR_DROPS = jsonl {"line": id} of the turn lines it dropped; turn lines it never saw are excluded too).
# Env: ANCHOR_DROPS, B1 (B1 run dir), EPOCHS (6), HIST (128), BATCH (512), HOURS (7), MAX_DPH (0.40), MIN_GPU_RAM (24),
#      SMOKE=1 (tiny, ~30 min).
cd "$(dirname "$0")/../.."
B1=${B1:-runs/stage-b1-20261013-0008}
for f in data/cache/base_r3/best.pt data/cache/base_r2/best.pt data/models/0546_26550M.tflite \
         data/cards/emb/qwen3-0.6b-d512.npz data/cards/emb/strings-qwen3-0.6b-d512.npz data/cards/lua_desc.npz \
         "$B1/clean/best.pt" "$B1/clean/metrics.jsonl"; do
  [ -f "$f" ] || { echo "missing $f"; exit 3; }
done
[ -s "${ANCHOR_DROPS:-}" ] || { echo "set ANCHOR_DROPS (turn-anchor drops jsonl)"; exit 3; }
mkdir -p data/cache/bv1 data/cache/b1_games
cp "$B1/clean/best.pt" "$B1/clean/metrics.jsonl" data/cache/bv1/
for r in bclean ftswv base; do
  ls "$B1/$r"/games-*.jsonl > /dev/null 2>&1 || { echo "no B1 games in $B1/$r"; exit 3; }
  mkdir -p "data/cache/b1_games/$r"; cp "$B1/$r"/games-*.jsonl "data/cache/b1_games/$r/"
done
# only anchored turn lines train (as B1): lines the anchor never saw are excluded with its drops
EX="$(dirname "$ANCHOR_DROPS")/b_excl.jsonl"
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
N=stage-b-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
D=(data/demos/teacher data/demos/dagger data/demos/gold data/demos/verified data/demos/expanded data/demos/claude
   data/demos/planned data/demos/planned-s1 data/demos/planned-s2 data/demos/planned-s3 data/demos/planned-s4
   data/demos/guides-strict-d data/demos/turns "$(dirname "$EX")" data/cache/audit data/cards/emb data/cards/lua_desc.npz
   decks data/models data/cache/base_r3 data/cache/base_r2 data/cache/bv1 data/cache/b1_games)
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "ANCHOR_DROPS=$EX EPOCHS=${EPOCHS:-6} HIST=${HIST:-128} BATCH=${BATCH:-512} SMOKE=${SMOKE:-0} GA_SEEDS=${GA_SEEDS:-24} bash scripts/vast_stage_b.sh $N" \
  --hours "${HOURS:-7}" --max-dph "${MAX_DPH:-0.40}" --min-cpus "${MIN_CPUS:-32}" --min-gpu-ram "${MIN_GPU_RAM:-24}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64}" \
  --data "${D[@]}" > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
