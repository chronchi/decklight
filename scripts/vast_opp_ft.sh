#!/usr/bin/env bash
# Fine-tune P on opponent-turn window labels (NOTEBOOK 2026-11-04 16:00; user: "Go ahead, let's do fine tune ... the
# focus is now on P"), then the pre-registered gates, fail-fast in this order:
#   1. anchor games: P vs P self-play (scripts/gen_selfplay.py, kind pgame), ANCHOR_GAMES;
#   2. shards: the window records (WIN_DIR, from scripts/vast_opp_windows.sh; kind window) + the anchor games;
#   3. train_bc from P: CE on both kinds, --kind-frac window=WREP, --kl-init KLI, EPOCHS, LR, --select-on last;
#   4. learned check: on the training windows labelled "search", the fine-tune must pick the label in >= 50%;
#   5. G1 scripts/audit/window_alt.py on held-out lines (cand's changed moves played out vs P's);
#   6. G2 the combo-line card (scripts/audit/line_eval.py, cand vs P, held-out lines, REPEATS);
#   7. G3 general play: scripts/eval_ckpt.py (default seeds) vs P and vs ygo-agent, fine-tune and P on the same deals.
# Env: WIN_DIR, EVAL_STARTS, CAND_CKPT (skip stages 1-4: gates only on this checkpoint), ANCHOR_GAMES (640), WREP (8), KLI (0.3), EPOCHS (2), LR (1e-4), CAP (90m per stage),
#      SMOKE=1 (tiny everything). Usage: bash scripts/vast_opp_ft.sh <name>. Do NOT run on the laptop (vast GPU only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; OUT="runs/$NAME"; mkdir -p "$OUT/ms" "$OUT/wa" "$OUT/line"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
export HF_HOME="$PWD/data/cache/hf"
P=data/cache/forge_r1/best.pt
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch || { alert "uv sync failed"; exit 3; }
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
. scripts/cores.sh
J=$(box_cores)
GMB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RMB=$(free -m | awk '/Mem/{print $2}')
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
T0=$SECONDS
tick() { echo "$(date +%T) $1 $((SECONDS - T0)) s" | tee -a "$OUT/PROGRESS"; }
SM=${SMOKE:-0}
AG=${ANCHOR_GAMES:-640}; EP=${EPOCHS:-2}; MXS=(); REP=${REPEATS:-4}; ES=(); CAP=${CAP:-90m}
[ "$SM" = 1 ] && { AG=16; EP=1; MXS=(--max-starts 1); REP=1; ES=(--eval-seeds 1 --ga-seeds 1); }

if [ -n "${CAND_CKPT:-}" ]; then  # gates only, on an existing fine-tune (a confirmation on fresh held-out starts)
  CAND=$CAND_CKPT; [ -s "$CAND" ] || { alert "CAND_CKPT $CAND missing"; exit 3; }
  tick "gates only: cand $CAND, starts $EVAL_STARTS"
else
# 1. anchor games: P vs P, GPU policies, 2 per process (student + league opponent)
NP=$(python3 -c "print(max(1, min($J, $GMB // 1500, $RMB // 3000)))")
PER=$(( (AG + NP - 1) / NP ))
tick "anchor games: $NP procs x $PER (cores $J, gpu ${GMB}MB, ram ${RMB}MB)"
GP=()
for p in $(seq 0 $((NP - 1))); do
  timeout "$CAP" uv run --no-sync python scripts/gen_selfplay.py --ckpt "$P" --out "$OUT/pgames/a-p$p" --games "$PER" \
    --temp 1.0 --league "R=$P:1" --seed $((52000 + 10000 * p)) --threads 1 --device cuda > "$OUT/gen-$p.txt" 2>&1 \
    || echo "gen $p exit $?" >> "$OUT/gen_fail.txt" &
  GP+=($!)
done
wait "${GP[@]}"
NG=$(cat "$OUT"/pgames/*/*.jsonl 2>/dev/null | wc -l)
tick "anchor games done: $NG"
[ "$NG" -ge $(( AG * 9 / 10 )) ] || { alert "anchor games $NG < 90% of $AG: $(cat "$OUT/gen_fail.txt" 2>/dev/null | head -3)"; exit 4; }
[ -s "$OUT/gen_fail.txt" ] && alert "anchor game workers failed (kept $NG games): $(head -3 "$OUT/gen_fail.txt" | tr '\n' ' ')"

# 2. shards
uv run --no-sync python scripts/build_bc_shards.py --glob "$WIN_DIR/*.jsonl" "$OUT/pgames/*/*.jsonl" \
  --out data/cache/bc_shards_oppft --jobs "$J" > "$OUT/shards.txt" 2>&1 || { alert "shards: $(tail -1 "$OUT/shards.txt")"; exit 4; }
tick "shards $(grep -m1 '^samples' "$OUT/shards.txt")"

# 3. fine-tune
HOLD="kewl-tune $(tr '\n' ' ' < decks/pool/HOLDOUT)"
# shellcheck disable=SC2086
uv run --no-sync python scripts/train_bc.py --shards data/cache/bc_shards_oppft --emb data/cards/emb/qwen3-0.6b-d512.npz \
  --out "$OUT/ft" --init "$P" --epochs "$EP" --batch 512 --lr "${LR:-1e-4}" --holdout $HOLD --select-on last \
  --kinds window pgame --require-kinds window pgame --kind-frac "window=${WREP:-8}" --kl-init "${KLI:-0.3}" \
  > "$OUT/train.txt" 2>&1 || { alert "train_bc exit $?: $(tail -1 "$OUT/train.txt")"; exit 4; }
CAND="$OUT/ft/best.pt"
tick "trained: $(grep -c . "$OUT/ft/metrics.jsonl") metric rows; last $(tail -1 "$OUT/ft/metrics.jsonl" | cut -c1-300)"

# 4. learned check (training windows, so it only shows the labels were fitted; the gates below are held out)
uv run --no-sync python - "$WIN_DIR" "$CAND" "$P" <<'PY' > "$OUT/learned.txt" 2>&1
import json, sys
from pathlib import Path
import numpy as np
from decklight.demos.dataset import iter_samples
from decklight.engine.duel import Engine
from decklight.eval.bc_policy import BCPolicy
from decklight.model.bc_data import equivalent_mask
from decklight.search.turn1 import _batch
d, emb = Path(sys.argv[1]), Path("data/cards/emb/qwen3-0.6b-d512.npz")
info = {}
for f in sorted(d.glob("*.jsonl")):
    for x in f.read_text().splitlines():
        r = json.loads(x)
        info.update({(r["line_id"], int(k)): v["kind"] for k, v in r["window_info"].items()})
pols = {"ft": BCPolicy(Path(sys.argv[2]), emb, device="cuda"), "P": BCPolicy(Path(sys.argv[3]), emb, device="cuda")}
hit = {(m, kd): [0, 0] for m in pols for kd in ("agree", "search", "p")}
for obs, act, w, meta in iter_samples(Engine(), d):
    kd = info.get((meta["line_id"], meta["k"]))
    if kd is None or not w:
        continue
    for m, pol in pols.items():
        z = pol.logits(_batch([obs]))[0]
        z = np.where(np.asarray(obs["action_mask"]).astype(bool), z[: len(obs["action_mask"])], -np.inf)
        eq = equivalent_mask(np.asarray(obs["actions"]), act, np.asarray(obs["action_mask"]).astype(bool))
        hit[(m, kd)][0] += bool(eq[int(np.argmax(z))])  # a copy of the labelled card counts (training uses the set)
        hit[(m, kd)][1] += 1
print({f"{m}/{kd}": f"{a}/{b}" for (m, kd), (a, b) in hit.items()})
a, b = hit[("ft", "search")]
sys.exit(0 if b and a / b >= 0.5 else 1)
PY
LC=$?
tick "learned: $(tail -1 "$OUT/learned.txt")"
if [ "$LC" -ne 0 ]; then  # the smoke's 1-epoch model is not expected to learn: alert, but go on to exercise the gates
  alert "fine-tune did not learn the search labels: $(tail -1 "$OUT/learned.txt")"; [ "$SM" = 1 ] || exit 4
fi

fi

run_shards() { # <script> <outdir> <extra args...>: CPU shards over every core, then merge
  local sc=$1 od=$2; shift 2
  local LJ=$(( J < RMB * 8 / 10 / 1100 ? J : RMB * 8 / 10 / 1100 )); [ "$LJ" -lt 1 ] && LJ=1
  local PS=()
  for i in $(seq 0 $((LJ - 1))); do
    timeout "$CAP" uv run --no-sync python "$sc" --starts "$EVAL_STARTS" "$@" "${MXS[@]}" --shard "$i/$LJ" \
      --device cpu --out "$od/s$i.json" > "$od/s$i.txt" 2>&1 || echo "shard $i exit $?" >> "$od/fail.txt" &
    PS+=($!)
  done
  wait "${PS[@]}"
  if [ -s "$od/fail.txt" ]; then  # a gate on a subset of the held-out lines is not the pre-registered gate
    alert "$sc shards: $(sort "$od/fail.txt" | uniq -c | tr '\n' ' ' | cut -c1-200)"; return 1
  fi
  uv run --no-sync python "$sc" --starts "$EVAL_STARTS" --merge "$od"/s*.json --out "$od/summary.json" > "$od/merge.txt" 2>&1
}

# 5. G1: changed moves on held-out windows, played out
run_shards scripts/audit/window_alt.py "$OUT/wa" --cand "$CAND" || { alert "G1 merge: $(tail -1 "$OUT/wa/merge.txt")"; exit 4; }
tick "G1 window_alt: $(tail -1 "$OUT/wa/merge.txt")"
uv run --no-sync python - "$OUT/wa/summary.json" <<'PY' >> "$OUT/PROGRESS" || { alert "G1 FAIL (see PROGRESS)"; [ "$SM" = 1 ] || exit 5; }  # the smoke runs every gate
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
ok = s["comparable"] >= 30 and s["starts_better"] > s["starts_worse"] and s["starts_p"] < 0.05
print(f"G1 {'PASS' if ok else 'FAIL'}: comparable changed windows {s['comparable']} (>= 30) over {s['starts_changed']} "
      f"starts; starts better {s['starts_better']} worse {s['starts_worse']} p {s['starts_p']} (< 0.05); windows: games "
      f"better {s['games_better']} worse {s['games_worse']}, field fewer {s['field_fewer']} more {s['field_more']}")
sys.exit(0 if ok else 1)
PY

# 6. G2: combo-line card, cand vs P (paired)
run_shards scripts/audit/line_eval.py "$OUT/line" --ckpt "$CAND" --vs-ckpt "$P" --repeats "$REP" \
  || { alert "G2 merge: $(tail -1 "$OUT/line/merge.txt")"; exit 4; }
tick "G2 combo-line card (cand - P): $(tail -1 "$OUT/line/merge.txt" | cut -c1-400)"
uv run --no-sync python - "$OUT/line/summary.json" <<'PY' >> "$OUT/PROGRESS" || { alert "G2 FAIL (see PROGRESS)"; [ "$SM" = 1 ] || exit 5; }  # the smoke runs every gate
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
ok = s["diff"] > 0 and s["p_one_sided"] is not None and s["p_one_sided"] < 0.05
print(f"G2 {'PASS' if ok else 'FAIL'}: cand - P {s['diff']} (se {s['se']}, pairs {s['pairs']}), p {s['p_one_sided']} (< 0.05)")
sys.exit(0 if ok else 1)
PY

# 7. G3: general play, same deals for the fine-tune and P
EVN=$(uv run --no-sync python - <<'PY'
import random
from pathlib import Path
hold = set(Path("decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
pool = sorted(Path("decks/pool").glob("*.ydk"))
r = random.Random(1234)
print(",".join(r.sample([p.stem for p in pool if p.stem not in hold], 4) + r.sample(sorted(hold & {p.stem for p in pool}), 4)))
PY
)
for tag in ft P; do
  ck=$CAND; [ "$tag" = P ] && ck=$P
  uv run --no-sync python scripts/eval_ckpt.py --ckpt "$ck" --init "$P" --eval-names "$EVN" "${ES[@]}" \
    --out "$OUT/ms/$tag.jsonl" --games-dir "$OUT/ms/$tag" > "$OUT/ms/$tag.txt" 2>&1 \
    || { alert "G3 eval $tag: $(tail -1 "$OUT/ms/$tag.txt")"; exit 4; }
done
uv run --no-sync python - "$OUT" <<'PY' >> "$OUT/PROGRESS" || { alert "G3 FAIL (see PROGRESS)"; [ "$SM" = 1 ] || exit 5; }  # the smoke runs every gate
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
ev = {t: json.loads((o / f"ms/{t}.jsonl").read_text().splitlines()[-1])["eval"] for t in ("ft", "P")}
f, p = ev["ft"], ev["P"]
d = f["vs_ygoagent"]["wr"] - p["vs_ygoagent"]["wr"]
print(f"G3 vs P {f['vs_bc']['wr']} n={f['vs_bc']['n']} {f['vs_bc']['ci']}; vs ygo-agent ft {f['vs_ygoagent']['wr']} "
      f"P {p['vs_ygoagent']['wr']} (n {f['vs_ygoagent']['n']}, diff {100 * d:+.1f} pp)")
ok = f["vs_bc"]["wr"] >= 0.49 and d >= -0.015
print(f"G3 {'PASS' if ok else 'FAIL'}: vs P >= 0.49 and vs ygo-agent >= P - 1.5 pp")
sys.exit(0 if ok else 1)
PY
tick "done"
exit 0
