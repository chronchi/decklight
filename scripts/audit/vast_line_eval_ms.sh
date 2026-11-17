#!/usr/bin/env bash
# Window PPO milestone audit on the training box (NOTEBOOK 2026-11-08; user: "evaluate well what's happening in terms
# of ... interrupting ... the policy interrupting the opponent after setting up the board in 1st turn"): for snapshot
# IT of run X: export, the window gate (--reuse), and line_eval A vs X on the held-out lines (argmax + sampled x2) and
# on the heavy turn-2 training starts (argmax), sharded over SH processes. Usage (from the repo on the box):
#   bash scripts/audit/vast_line_eval_ms.sh <run dir: runs/tsoj-arms/<name>> IT [SH=12]
set -uo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" OMP_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0
O="$1"; IT=$(printf %04d "$2"); SH="${3:-12}"; T="$O/train"; M="$T/ms-$IT"; mkdir -p "$M"
W=data/cache/window_ppo; A=$W/A/policy/best.pt; X="$T/export-$IT/policy/best.pt"
alert() { echo "ALERT $*" | tee -a "$O/ALERT"; }
while [ ! -s "$T/X/snap-$IT.pt" ]; do sleep 30; done; sleep 20
[ -s "$X" ] || uv run --no-sync python scripts/tsoj/export_rl.py "$T/X" "$T/export-$IT" --init $A --value-init $W/A/value/best.pt \
  --ckpt "$T/X/snap-$IT.pt" > "$M/export.log" 2>&1 || { alert "ms $IT export failed"; exit 3; }
echo "ms $IT start $(date +%T)" | tee -a "$O/PROGRESS"
[ -s "$T/wg-$IT.json" ] || OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 uv run --no-sync python -u scripts/audit/window_gate.py \
  --starts data/cache/line_starts/starts-v2.eval.jsonl --cand "$X" --reuse $W/wg/s[0-9]*.json --out "$T/wg-$IT.json" > "$T/wg-$IT.txt" 2>&1 &
run() {  # run TAG STARTS TEMP REPEATS
  for i in $(seq 0 $((SH - 1))); do
    OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 uv run --no-sync python -u scripts/audit/line_eval.py --starts "$2" --ckpt $A --vs-ckpt "$X" \
      --temp "$3" --repeats "$4" --shard "$i/$SH" --out "$M/$1-s$i.json" > "$M/$1-s$i.log" 2>&1 &
  done; wait
  python3 scripts/audit/line_eval_split.py "$M"/$1-s*.json --out "$M/$1.split.json" > "$M/$1.split.txt" 2>&1 || alert "ms $IT $1 split failed: $(tail -2 "$M/$1.split.txt")"
  echo "ms $IT $1: $(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
for k,v in d.items(): print(k, 'pairs', v['pairs'], 'X-A', v['B_minus_A'], '+-', v['se'], '| won A/X', v['A']['won'], v['B']['won'], '| activated', v['A']['activated'], v['B']['activated'], '| acts by location A/X', {k[5:]: (v['A'].get(k, 0), v['B'].get(k, 0)) for k in sorted(set(v['A']) | set(v['B'])) if k.startswith('acts_')}, '| completed', v['A']['line_completed'], v['B']['line_completed'], '| lp_lost', v['A']['lp_lost'], v['B']['lp_lost'], end=' ;; ')
" "$M/$1.split.json")" | tee -a "$O/PROGRESS"
}
run heldout-argmax data/cache/line_starts/starts-v2.eval.jsonl 0 1
run t2heavy-argmax data/cache/line_starts/starts-v2-heavy.second.jsonl 0 1
run heldout-sampled data/cache/line_starts/starts-v2.eval.jsonl 1 2
wait
[ -s "$T/wg-$IT.json" ] && echo "WG it$2: $(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['summary']))" "$T/wg-$IT.json")" | tee -a "$O/PROGRESS"
echo "ms $IT done $(date +%T)" | tee -a "$O/PROGRESS"
