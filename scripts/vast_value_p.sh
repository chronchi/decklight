#!/usr/bin/env bash
# P's value net + leaf choice + P with search (user 2026-10-13: "P's critic, value-only RL (Recommended)"; NOTEBOOK 23:30).
#   1. value head: train_rl from P with every iteration a value warm-up (the policy and trunk stay frozen; only
#      value_head trains), on P's own league rollouts, VITERS iterations. P's critic was trained on the same trunk
#      during PPO, but stage P deleted it with last.pt. Exported fp16 to data/cache/value_p (+ metrics args value: true);
#      exit 4 when any policy weight moved.
#   2. leaf choice, offline (plan-v2): scripts/audit/leaf_compare.py on held-out Pc games (P raw vs ygo-agent, seeds
#      3000+): value_p vs value_ppo3 on turn-start states. The lower MSE wins unless its ECE is worse by > 0.02;
#      inconclusive (|dMSE| <= 1 SE) -> the lower point-estimate MSE (user 2026-10-14, budget), one search arm.
#   3. P raw and P + hppo (honest k4 s4 h1, det-tries 32, skip, conf-skip 0.9: the accepted product's search) with the
#      chosen value(s) vs ygo-agent on 1,488 FRESH deals (deal seeds 5000+; F keeps 4000+), paired; principles on hppo.
# Env: VITERS (150), GA_SEEDS (6), VAR_CAP (240m), BOTH=1 (both values get a search arm), SMOKE=1 (4 value iterations, 40 leaf games, 1 seed).
# Usage: bash scripts/vast_value_p.sh <name>. Do NOT run on the laptop (vast GPU only).
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
P=data/cache/P/best.pt; VP=data/cache/value_p; PPO3=data/cache/value_ppo3/best.pt
[ -s "$P" ] && [ -s data/cache/P/metrics.jsonl ] && [ -s "$PPO3" ] || { alert "missing P or value_ppo3"; exit 3; }
VITERS=${VITERS:-150}; LEAF_GAMES=3000; GS=${GA_SEEDS:-6}
[ "${SMOKE:-0}" = 1 ] && { VITERS=4; LEAF_GAMES=40; GS=1; }

# 1. value head on P's frozen trunk (warm-up = value_head only, train_rl vopt)
uv run --no-sync python scripts/train_rl.py --init "$P" --out "$OUT/rl" --device cuda --iters "$VITERS" \
  --value-warmup "$VITERS" --eval-every 100000 --hours 3 > "$OUT/train.txt" 2>&1 &
TP=$!
( seen=0; while kill -0 $TP 2>/dev/null; do  # relay the value loss every 25 iterations (first milestone: finite, not rising)
    sleep 30
    n=$(grep -c '"vf"' "$OUT/rl/metrics.jsonl" 2>/dev/null); n=${n:-0}
    if [ "$n" -ge $((seen + 25)) ] || { [ "$seen" = 0 ] && [ "$n" -ge 5 ]; }; then
      echo "$(date +%T) value it $n: vf $(grep '"vf"' "$OUT/rl/metrics.jsonl" | tail -5 | grep -o '"vf": [0-9.na]*' | tr '\n' ' ')" >> "$OUT/PROGRESS"; seen=$n
    fi
  done ) &
wait $TP; rc=$?
grep -h "^ALERT\|Traceback" "$OUT/train.txt" | head -5 >> "$OUT/ALERT"
[ $rc -eq 0 ] && [ -s "$OUT/rl/last.pt" ] || { alert "value training failed (exit $rc): $(tail -1 "$OUT/train.txt")"; exit 4; }
tick "value head $VITERS iterations"
uv run --no-sync python - "$OUT" "$P" "$VP" "$VITERS" <<'PY' >> "$OUT/PROGRESS" || { alert "value export failed"; exit 4; }
import json, sys
from pathlib import Path
import numpy as np, torch
out, p, vp, iters = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
ck = torch.load(out / "rl/last.pt", map_location="cpu", weights_only=False)["model"]
ref = torch.load(p, map_location="cpu")
ref = ref.get("model", ref)
moved = [k for k, v in ref.items() if not torch.equal(ck[k].half() if ck[k].is_floating_point() else ck[k], v)]
if moved:
    sys.exit(f"ALERT policy weights moved during value-only training: {moved[:3]} ({len(moved)} tensors)")
vk = [k for k in ck if k.startswith("value_head.")]
if not vk:
    sys.exit("ALERT last.pt has no value_head")
vp.mkdir(parents=True, exist_ok=True)
torch.save({k: v.half() if v.is_floating_point() else v for k, v in ck.items()}, vp / "best.pt")
args = next(json.loads(x)["args"] for x in (p.parent / "metrics.jsonl").read_text().splitlines() if '"args"' in x)
(vp / "metrics.jsonl").write_text(json.dumps({"epoch": -1, "init": f"{out}/rl/last.pt (value-only, {iters} iterations on P)",
                                              "value": True, "args": {**args, "value": True}}) + "\n")
vf = [json.loads(x)["vf"] for x in (out / "rl/metrics.jsonl").read_text().splitlines() if '"vf"' in x]
print(f"value export: {len(vk)} value_head tensors -> {vp}/best.pt; policy unchanged; vf first5 {np.mean(vf[:5]):.4f} "
      f"last20 {np.mean(vf[-20:]):.4f}")
PY
cp -r "$VP" "$OUT/value_p"

# 2. leaf choice on held-out Pc games
uv run --no-sync python scripts/audit/leaf_compare.py --games data/cache/pc_games/p --value "new=$VP/best.pt" \
  --value "ppo3=$PPO3" --max-games "$LEAF_GAMES" --device cuda --out "$OUT/leaf.json" > "$OUT/leaf.txt" 2>&1 \
  || { alert "leaf_compare failed: $(tail -1 "$OUT/leaf.txt")"; exit 4; }
sed "s/^/$(date +%T) /" "$OUT/leaf.txt" | tail -4 >> "$OUT/PROGRESS"
LEAF=$(grep '^leaf:' "$OUT/leaf.txt" | awk '{print $2}')
# a tie takes the lower point-estimate MSE, one arm (user 2026-10-14: "Tie -> lower MSE, one arm (Recommended)";
# credit $X does not cover two 1,488-deal arms)
if [ "$LEAF" = inconclusive ]; then
  LEAF=$(uv run --no-sync python -c "import json; d=json.load(open('$OUT/leaf.json'))['turn_start_dmse']['dmse']; print('new' if d < 0 else 'ppo3')")
  echo "$(date +%T) leaf tie -> lower MSE: $LEAF" >> "$OUT/PROGRESS"
fi
case "$LEAF" in new) ARMS="new";; ppo3) ARMS="ppo3";; *) alert "no leaf verdict"; exit 4;; esac
# BOTH=1: a search arm for each value whatever the verdict, chosen first (user 2026-10-14: "I'll want both comparisons
# actually to make an informed decision if the new value net is trained well"; 992 deals each)
[ "${BOTH:-0}" = 1 ] && { [ "$ARMS" = new ] && ARMS="new ppo3" || ARMS="ppo3 new"; }

# 3. P raw + P with search on fresh deals
H="--honest 1 --det-tries 32 --det-fallback skip --horizon 1 --claim 1"
V="raw|--opp ygoagent --raw 1 --ga-seeds $GS"
for a in $ARMS; do
  [ "$a" = new ] && VAL="$VP/best.pt" || VAL="$PPO3"
  V="$V;hppo-$a|--opp ygoagent --ga-seeds $GS $H --value $VAL"
done
FIRST=$(echo "$ARMS" | awk '{print $1}'); [ "$FIRST" = new ] && FV="$VP/best.pt" || FV="$PPO3"
CKPT="$P" SM_ARGS="--k 4 --samples 4 --conf-skip 0.9 --deal-seed0 5000" VARIANTS="$V" \
  PROBE_ARGS="--opp ygoagent --ga-seeds $GS $H --value $FV" PROBE_NPROC=${PROBE_NPROC:-40} VAR_CAP="${VAR_CAP:-240m}" \
  bash scripts/vast_search_match.sh "$NAME"
tick "card"
for a in $ARMS; do
  uv run --no-sync python scripts/audit/paired_sm.py "$OUT/raw" "$OUT/hppo-$a" > "$OUT/paired-$a.txt" 2>&1 \
    || alert "paired_sm $a: $(tail -1 "$OUT/paired-$a.txt")"
  echo "$(date +%T) paired hppo-$a - raw: $(sed -n 1,4p "$OUT/paired-$a.txt" | tr '\n' ' ')" >> "$OUT/PROGRESS"
  for s in s ya; do
    uv run --no-sync python scripts/audit/principles.py --name "hppo-$a-$s" --games "$OUT/hppo-$a" --seat $s \
      --out "$OUT/principles-hppo-$a-$s.json" > "$OUT/principles-hppo-$a-$s.txt" 2>&1 || alert "principles hppo-$a $s failed"
  done
  uv run --no-sync python - "$OUT" "$a" <<'PY' >> "$OUT/PROGRESS"
import json, sys
o, a = sys.argv[1], sys.argv[2]
s, ya = (json.load(open(f"{o}/principles-hppo-{a}-{x}.json")) for x in ("s", "ya"))
print(f"hppo-{a} behaviour: own-T1 HT games {s['lines_with_own_t1_ht']} (gate <= 1.5 x ygo-agent "
      f"{ya['lines_with_own_t1_ht']}); interrupts {s['interrupt_opp']['rate']} vs ygo-agent {ya['interrupt_opp']['rate']}")
PY
done
if [ "${BOTH:-0}" = 1 ] && [ -d "$OUT/hppo-new" ] && [ -d "$OUT/hppo-ppo3" ]; then  # the two values head to head, same deals
  uv run --no-sync python scripts/audit/paired_sm.py "$OUT/hppo-ppo3" "$OUT/hppo-new" > "$OUT/paired-new-vs-ppo3.txt" 2>&1 \
    || alert "paired_sm new vs ppo3: $(tail -1 "$OUT/paired-new-vs-ppo3.txt")"
  echo "$(date +%T) paired hppo-new - hppo-ppo3: $(sed -n 1,4p "$OUT/paired-new-vs-ppo3.txt" | tr '\n' ' ')" >> "$OUT/PROGRESS"
fi
