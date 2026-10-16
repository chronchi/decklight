#!/usr/bin/env bash
# value_p2: continue value-only RL from value_p (user 2026-10-15: "Go with More value-only RL on value_p, with more
# iterations only for now"; NOTEBOOK 2026-10-16 00:30). P's policy and trunk stay frozen.
#   1. value head: train_rl --init P --value-init value_p, every iteration a value warm-up (value_head only, its own
#      optimizer at --value-lr 1e-4: value_p was trained at a constant 3e-4 and its vf drifted up at the end), on P's
#      own league rollouts (no ygo-agent), VITERS iterations. Milestone: the first 5 iterations' vf must start at
#      value_p's level (< VF_START_MAX), else the warm start did not apply -> stop. Exit 4 when a policy weight moved.
#   2. offline sanity on held-out Pc games (leaf_compare, value_p2 vs value_p): if value_p2's turn-start MSE is worse by
#      more than 2 SE, stop before the card (offline MSE does not pick the better search leaf, NOTEBOOK 2026-10-14
#      17:10, so it is only a stop for a clearly broken head).
#   3. card: P + hppo (the product's search) with value_p2 and with value_p vs ygo-agent on the SAME fresh deals
#      (deal seeds 9000+, first use), GA_SEEDS per pair (6 = 1,488 deals), paired value_p2 - value_p, both seats.
# Env: VITERS (450), GA_SEEDS (6), VAR_CAP (170m), VF_START_MAX (0.12), SMOKE=1 (4 iterations, 40 leaf games, 1 seed).
# Usage: bash scripts/vast_value_p2.sh <name>. Do NOT run on the laptop (vast GPU only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export HF_HOME="$PWD/data/cache/hf"
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }
P=data/cache/P/best.pt; VP=data/cache/value_p/best.pt; VP2=data/cache/value_p2
[ -s "$P" ] && [ -s data/cache/P/metrics.jsonl ] && [ -s "$VP" ] || { alert "missing P or value_p"; exit 3; }
VITERS=${VITERS:-450}; LEAF_GAMES=3000; GS=${GA_SEEDS:-6}
[ "${SMOKE:-0}" = 1 ] && { VITERS=4; LEAF_GAMES=40; GS=1; }

# 1. value head, warm from value_p, on P's frozen trunk
uv run --no-sync python scripts/train_rl.py --init "$P" --value-init "$VP" --value-lr 1e-4 --out "$OUT/rl" \
  --device cuda --iters "$VITERS" --value-warmup "$VITERS" --eval-every 100000 --hours 3 > "$OUT/train.txt" 2>&1 &
TP=$!
( seen=0; while kill -0 $TP 2>/dev/null; do  # relay vf every 25 iterations; stop if the warm start did not apply
    sleep 30
    n=$(grep -c '"vf"' "$OUT/rl/metrics.jsonl" 2>/dev/null); n=${n:-0}
    if [ "$seen" = 0 ] && [ "$n" -ge 5 ]; then
      v=$(grep '"vf"' "$OUT/rl/metrics.jsonl" | head -5 | grep -o '"vf": [0-9.]*' | awk '{s+=$2} END {printf "%.4f", s/NR}')
      echo "$(date +%T) milestone: vf first-5 mean $v (value_p's run ended at ~0.079; max ${VF_START_MAX:-0.12})" >> "$OUT/PROGRESS"
      if awk -v v="$v" -v m="${VF_START_MAX:-0.12}" 'BEGIN {exit !(v > m)}'; then
        alert "vf first-5 mean $v > ${VF_START_MAX:-0.12}: the value_p warm start did not apply"; kill $TP; break
      fi
      seen=5
    elif [ "$n" -ge $((seen + 25)) ]; then
      echo "$(date +%T) value it $n: vf $(grep '"vf"' "$OUT/rl/metrics.jsonl" | tail -5 | grep -o '"vf": [0-9.na]*' | tr '\n' ' ')" >> "$OUT/PROGRESS"; seen=$n
    fi
  done ) &
wait $TP; rc=$?
grep -h "^ALERT\|Traceback" "$OUT/train.txt" | head -5 >> "$OUT/ALERT"
[ $rc -eq 0 ] && [ -s "$OUT/rl/last.pt" ] || { alert "value training failed (exit $rc): $(tail -1 "$OUT/train.txt")"; exit 4; }
tick "value head $VITERS iterations"
uv run --no-sync python - "$OUT" "$P" "$VP" "$VP2" "$VITERS" <<'PY' >> "$OUT/PROGRESS" || { alert "value export failed"; exit 4; }
import json, sys
from pathlib import Path
import numpy as np, torch
out, p, vp, vp2, iters = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
ck = torch.load(out / "rl/last.pt", map_location="cpu", weights_only=False)["model"]
ref = torch.load(p, map_location="cpu")
ref = ref.get("model", ref)
moved = [k for k, v in ref.items() if not torch.equal(ck[k].half() if ck[k].is_floating_point() else ck[k], v)]
if moved:
    sys.exit(f"ALERT policy weights moved during value-only training: {moved[:3]} ({len(moved)} tensors)")
vk = [k for k in ck if k.startswith("value_head.")]
old = torch.load(vp, map_location="cpu")
old = old.get("model", old)
if not vk or all(torch.equal(ck[k].half(), old[k]) for k in vk):
    sys.exit("ALERT value_head missing or identical to value_p's")
vp2.mkdir(parents=True, exist_ok=True)
torch.save({k: v.half() if v.is_floating_point() else v for k, v in ck.items()}, vp2 / "best.pt")
args = next(json.loads(x)["args"] for x in (p.parent / "metrics.jsonl").read_text().splitlines() if '"args"' in x)
(vp2 / "metrics.jsonl").write_text(json.dumps({"epoch": -1, "init": f"{out}/rl/last.pt (value-only, {iters} more "
                                               "iterations from value_p, lr 1e-4)", "value": True,
                                               "args": {**args, "value": True}}) + "\n")
vf = [json.loads(x)["vf"] for x in (out / "rl/metrics.jsonl").read_text().splitlines() if '"vf"' in x]
print(f"value export: {len(vk)} value_head tensors -> {vp2}/best.pt; policy unchanged; vf first5 {np.mean(vf[:5]):.4f} "
      f"last20 {np.mean(vf[-20:]):.4f}")
PY
cp -r "$VP2" "$OUT/value_p2"

# 2. offline sanity on held-out Pc games
uv run --no-sync python scripts/audit/leaf_compare.py --games data/cache/pc_games/p --value "p2=$VP2/best.pt" \
  --value "p=$VP" --max-games "$LEAF_GAMES" --device cuda --out "$OUT/leaf.json" > "$OUT/leaf.txt" 2>&1 \
  || { alert "leaf_compare failed: $(tail -1 "$OUT/leaf.txt")"; exit 4; }
sed "s/^/$(date +%T) /" "$OUT/leaf.txt" | tail -4 >> "$OUT/PROGRESS"
uv run --no-sync python - "$OUT/leaf.json" <<'PY' || { alert "value_p2 clearly worse offline; card skipped"; exit 5; }
import json, sys
d = json.load(open(sys.argv[1]))["turn_start_dmse"]
print(f"turn-start dMSE (p2 - p) {d['dmse']:+.4f} SE {d['se']:.4f}")
sys.exit(1 if d["dmse"] > 2 * d["se"] else 0)
PY

# 3. card: the product's search with each value, same fresh deals
H="--honest 1 --det-tries 32 --det-fallback skip --horizon 1 --claim 1"
CKPT="$P" SM_ARGS="--k 4 --samples 4 --conf-skip 0.9 --deal-seed0 9000" \
  VARIANTS="hppo-p2|--opp ygoagent --ga-seeds $GS $H --value $VP2/best.pt;hppo-p|--opp ygoagent --ga-seeds $GS $H --value $VP" \
  PROBE_NPROC=${PROBE_NPROC:-40} VAR_CAP="${VAR_CAP:-170m}" bash scripts/vast_search_match.sh "$NAME"
tick "card"
uv run --no-sync python scripts/audit/paired_sm.py "$OUT/hppo-p" "$OUT/hppo-p2" > "$OUT/paired.txt" 2>&1 \
  || alert "paired_sm: $(tail -1 "$OUT/paired.txt")"
echo "$(date +%T) paired hppo-p2 - hppo-p: $(sed -n 1,4p "$OUT/paired.txt" | tr '\n' ' ')" >> "$OUT/PROGRESS"
for a in p2 p; do
  for s in s ya; do
    uv run --no-sync python scripts/audit/principles.py --name "hppo-$a-$s" --games "$OUT/hppo-$a" --seat $s \
      --out "$OUT/principles-hppo-$a-$s.json" > "$OUT/principles-hppo-$a-$s.txt" 2>&1 || alert "principles hppo-$a $s failed"
  done
done
