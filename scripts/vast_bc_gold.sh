#!/usr/bin/env bash
# Gold-game BC experiment (STATUS step 2, 2026-10-04): does training on the hand-played gold games help?
# Base = the best previous BC checkpoint, shipped in data/cache/bc_base/ (best.pt + metrics.jsonl).
# Variants, all with the same split (holdout decks + 25% of gold games in val) and best.pt picked by val gold accuracy:
#   zero/     base evaluated as is (epoch -1 record only)
#   ft-gold/  base fine-tuned on gold samples only
#   ft-mix/   base fine-tuned on everything, gold repeated
#   mix/      from scratch, recipe v2 + gold repeated
#   lines-gold/ from scratch, engine-verified combo lines + gold only: the noisy LLM games (game_plan / game_interrupt)
#             dropped (2026-10-04 user decision)
#   mix-l/    mix with a 3x larger model: d=384, 6 layers, 6 heads (~12M params vs ~4M; capacity axis, 2026-10-05)
#   (SKIP_SCRATCH=1 skips the three from-scratch runs)
# then playtests the base and the variant with the best val gold accuracy on the same decks/seeds.
# Env: BASE=<ckpt> (default data/cache/bc_base/best.pt), PLAY_ALL=1 playtests every variant in VARIANTS, PT_ARGS
# overrides the playtest flags, GOLD_GAMES=N trains on the first N gold games only (saturation curve; eval sets unchanged), PT_ZERO=0 skips the
# base model's playtest (it is the same in every job of a curve).
# Usage: bash scripts/vast_bc_gold.sh <name> [smoke]. smoke = gold shards only, 1 epoch, no scratch run, tiny playtest
# (ship --data data/demos for a real playtest: it samples decks that have verified lines; with only data/demos/gold it
# finds none and the match section fails, as in bcg-smoke-20261004-2142).
# Results in runs/<name>/ (pulled back by the launcher): <variant>/metrics.jsonl + best.pt, train-*.txt, playtest-*.txt,
# summary.txt.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; MODE="${2:-full}"; OUT="runs/$NAME"; mkdir -p "$OUT"
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
if [ "$Q" = "max" ]; then J=$(nproc); else J=$(( Q / P )); fi
[ "$J" -lt 2 ] && J=2
# CUDA torch straight away (the lock pins the CPU wheel: skip it instead of downloading both; 2026-10-05)
T0=$(date +%s)
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300  # 30 s default: a slow mirror timed out on nvidia-nccl (bcg-mem512-20261005-0926)
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
echo "deps install $(( $(date +%s) - T0 ))s" >> "$OUT/torch.txt"
{ nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; echo "jobs $J";
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; } > "$OUT/machine.txt" 2>&1
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA torch: abort" | tee -a "$OUT/torch.txt"; exit 3; }
# GPU/CPU usage every 30 s (is a better GPU worth it? 2026-10-05): gpu util %, gpu mem MiB, load average
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg)"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
T0=$SECONDS
tick() { echo "$1 $((SECONDS - T0)) s" >> "$OUT/timing.txt"; }
BASE="${BASE:-data/cache/bc_base/best.pt}"  # M1 teacher job: BASE=data/cache/rl_init/best.pt
if [ "$MODE" = "smoke" ]; then
  ONLY=(--only gold); EP_FT=1; EP_MIX=1; SKIP_SCRATCH=1; PT=(--decks 2 --hands 1 --seeds 1 --narrate 1 --samples 2 --ygoagent --selfplay 2); PT_CAP=12m
else
  ONLY=(); EP_FT=10; EP_MIX=3; SKIP_SCRATCH="${SKIP_SCRATCH:-0}"; PT=(--decks 8 --hands 3 --seeds 2 --narrate 2 --samples 4 --ygoagent --selfplay 10); PT_CAP=40m
fi
[ -n "${PT_ARGS:-}" ] && read -ra PT <<< "$PT_ARGS"
uv run --no-sync python scripts/build_bc_shards.py "${ONLY[@]}" --out data/cache/bc_shards --jobs "$J" > "$OUT/shards.txt" 2>&1
tick shards
# fail loud: a data source that mostly produced empty shards means a dataset bug; never train on it
if grep -q "^ALERT" "$OUT/shards.txt"; then grep -h "^ALERT" "$OUT/shards.txt" >> "$OUT/ALERT"; exit 3; fi
HOLD="kewl-tune $( [ -f decks/pool/HOLDOUT ] && tr '\n' ' ' < decks/pool/HOLDOUT )"
COMMON=(--shards data/cache/bc_shards --emb data/cards/emb/qwen3-0.6b-d512.npz --batch 512 --holdout $HOLD
        --val-frac-kind gold=0.25 gold_pass=0.25 --select-on val:gold)
[ -n "${GOLD_GAMES:-}" ] && COMMON+=(--gold-games "$GOLD_GAMES")
train() {  # <variant> <args...>
  local v="$1"; shift
  uv run --no-sync python scripts/train_bc.py "${COMMON[@]}" --out "$OUT/$v" "$@" > "$OUT/train-$v.txt" 2>&1
  local rc=$?
  rm -f "$OUT/$v/last.pt"
  # fail loud: ALERT lines reach the launcher log within one poll (src/ygo_zero/infra/vast.py run_detached)
  grep -q "^done in" "$OUT/train-$v.txt" || echo "ALERT train-$v died before 'done in' (exit $rc; OOM?)" >> "$OUT/ALERT"
  grep -h "^ALERT" "$OUT/train-$v.txt" >> "$OUT/ALERT"
  [ $rc -eq 0 ] || echo "FAILED train-$v exit $rc" >> "$OUT/timing.txt"
  tick "train-$v"
}
# VARIANTS: which trainings run (default: the gold-BC set). Memory sweep (ADR 0005): mix-h<H> = mix with the salient
# long history window H (train_bc --hist H); mix itself is the v1 baseline (32 entries, passes included).
VARIANTS=" ${VARIANTS:-zero ft-gold ft-mix mix lines-gold mix-l} "
want() { [[ "$VARIANTS" == *" $1 "* ]]; }
MIXK=(--kind-frac line_expanded=0.2 game_plan=2 game_interrupt=2 gold=20 gold_pass=10)
want zero && train zero --init "$BASE" --epochs 0  # the base's own metrics (skip in parallel per-variant jobs)
want ft-gold && train ft-gold --init "$BASE" --epochs "$EP_FT" --lr 1e-4 --batch 256 --kinds gold gold_pass
want ft-mix && train ft-mix --init "$BASE" --epochs "$EP_MIX" --lr 2e-4 \
  --kind-frac line_expanded=0.2 game_plan=1 game_interrupt=1 gold=20 gold_pass=10
[ "$SKIP_SCRATCH" = "1" ] || ! want mix || train mix --epochs 8 --lr 5e-4 --merge-last 4 \
  --kind-frac line_expanded=0.2 game_plan=2 game_interrupt=2 gold=20 gold_pass=10
[ "$SKIP_SCRATCH" = "1" ] || ! want lines-gold || train lines-gold --epochs 8 --lr 5e-4 --merge-last 4 \
  --kinds line line_improved line_expanded gold gold_pass --kind-frac line_expanded=0.2 gold=20 gold_pass=10
[ "$SKIP_SCRATCH" = "1" ] || ! want mix-l || train mix-l --epochs 8 --lr 3e-4 --merge-last 4 --d 384 --layers 6 --heads 6 \
  --kind-frac line_expanded=0.2 game_plan=2 game_interrupt=2 gold=20 gold_pass=10
# M1 (ADR 0006): fine-tune on ygo-agent self-play teacher games (data/demos/teacher, 3003 games on its 31 lists) as a
# minority mix (ft-teach) or a bigger share (ft-teach-h); gold stays repeated as in ft-mix
want ft-teach && train ft-teach --init "$BASE" --epochs "$EP_MIX" --lr 2e-4 --kind-frac line_expanded=0.2 game_plan=1 \
  game_interrupt=1 gold=20 gold_pass=10 teacher=0.3 teacher_pass=0.3
want ft-teach-h && train ft-teach-h --init "$BASE" --epochs "$EP_MIX" --lr 2e-4 --kind-frac line_expanded=0.2 game_plan=1 \
  game_interrupt=1 gold=20 gold_pass=10 teacher=1 teacher_pass=1
# M2 (ADR 0006): distill the turn-1 search teacher (data/demos/search, scripts/gen_search.py); teacher games excluded
want ft-search && train ft-search --init "$BASE" --epochs "$EP_MIX" --lr 2e-4 --kind-frac line_expanded=0.2 game_plan=1 \
  game_interrupt=1 gold=20 gold_pass=10 teacher=0 teacher_pass=0 search=3
# M3 mix (ADR 0006/0007): from BASE = ft-teach-h, teacher games + DAgger labels + mid-game window search (+ search)
# --select-on last: these mixes move the policy away from the LLM gold labels (more interruptions than the LLM's
# passes), so val:gold never beats the init and best.pt stayed the init in mix-20261005-1738 (NOTEBOOK)
# ft-dw: responses only (window_own excluded: own-turn targets use the heuristic score, ADR 0007 collapse risk)
want ft-dw && train ft-dw --init "$BASE" --epochs "$EP_MIX" --lr 2e-4 --select-on last --kind-frac line_expanded=0.2 game_plan=1 \
  game_interrupt=1 gold=20 gold_pass=10 teacher=1 teacher_pass=1 dagger=1 dagger_pass=1 window=1 window_own=0 search=0
# ft-all: every teacher, own-turn window targets and turn-1 search at a reduced share
want ft-all && train ft-all --init "$BASE" --epochs "$EP_MIX" --lr 2e-4 --select-on last --kind-frac line_expanded=0.2 game_plan=1 \
  game_interrupt=1 gold=20 gold_pass=10 teacher=1 teacher_pass=1 dagger=1 dagger_pass=1 window=1 window_own=0.5 search=1
# ADR 0007 round 1 (from BASE = the mix winner): every variant adds the scaled ygo-agent teacher games (subsampled to
# TFRAC per epoch); ft-swv / ft-swd add self-play window search labels (data/demos/sw-*, scripts/relabel_window.py)
# picked by the learned value at the end of the turn (value@h1) or by "won within 3 turns, then value" (done@h3,
# value@h3). ft-t13 is the control for both. Unused sw kinds must be 0 (kinds missing from --kind-frac keep all).
TFRAC=${TFRAC:-0.5}
R1=(--init "$BASE" --epochs "$EP_MIX" --lr 2e-4 --select-on last --kind-frac line_expanded=0.2 game_plan=1 game_interrupt=1
    gold=20 gold_pass=10 teacher="$TFRAC" teacher_pass="$TFRAC" dagger=1 dagger_pass=1 window=1 window_own=0 search=1)
# search=1 / window_own=0 from mix-20261005-2016 (NOTEBOOK 2026-10-05 23:20): without turn-1 search ft-dw lost combos (2.04 -> 1.54);
# ft-all (+ heuristic own-turn labels) lost on new-card decks (75.8% -> 66.7% vs greedy)
want ft-t13 && train ft-t13 "${R1[@]}" sw-v=0 sw-v_own=0 sw-d3=0 sw-d3_own=0
want ft-swv && train ft-swv "${R1[@]}" sw-v=2 sw-v_own=1 sw-d3=0 sw-d3_own=0 --require-kinds sw-v sw-v_own
want ft-swd && train ft-swd "${R1[@]}" sw-v=0 sw-v_own=0 sw-d3=2 sw-d3_own=1 --require-kinds sw-d3 sw-d3_own
# round 2 (ADR 0007): from ft-swv, round-1 + round-2 value@h1 labels (sw-v2: self-play of ft-swv under the loop guard,
# 4 rollout samples); ft-r2 keeps teacher at 0.5/epoch, ft-r2t halves it (all round-1 variants lost ~0.3 combo)
R2=(--init "$BASE" --epochs "$EP_MIX" --lr 2e-4 --select-on last --kind-frac line_expanded=0.2 game_plan=1 game_interrupt=1
    gold=20 gold_pass=10 dagger=1 dagger_pass=1 window=1 window_own=0 search=1 sw-v=2 sw-v_own=1 sw-v2=2 sw-v2_own=1
    sw-d3=0 sw-d3_own=0)
want ft-r2 && train ft-r2 "${R2[@]}" teacher=0.5 teacher_pass=0.5 --require-kinds sw-v2 sw-v2_own
want ft-r2t && train ft-r2t "${R2[@]}" teacher=0.25 teacher_pass=0.25 --require-kinds sw-v2 sw-v2_own
# round 3 (search plan v2, NOTEBOOK 2026-10-06 23:50): from base_r3 (PPO), round-2 mix + sw-b3 = oracle window search
# with the board score at horizon 1 on every unsure decision (S0's best teacher, 0.696 vs raw base_r3); own-turn
# labels weighted like windows this time (S0 searched both)
want ft-r3 && train ft-r3 "${R2[@]}" teacher=0.5 teacher_pass=0.5 sw-b3=2 sw-b3_own=2 --require-kinds sw-b3 sw-b3_own
for V in mix-h32 mix-h128 mix-h256 mix-h512 mix-h32k mix-h128k mix-h256k mix-h512k mix-k; do
  want "$V" || continue
  H=${V#mix-h}; H=${H%k}; [ "$V" = mix-k ] && H=0
  KN=(); [[ "$V" == *k ]] && KN=(--known)  # card-knowledge memory (ADR 0005 step B)
  [ "$MODE" = "smoke" ] && EPM=1 || EPM=8
  # activations grow with 1+160+16+H tokens: batch 512 OOMs a 12 GB card at H=512 (bcg-memsmoke-20261005-0900)
  case $H in 512) BH=(--batch 128 --lr 3e-4);; 256) BH=(--batch 256 --lr 4e-4);; *) BH=();; esac
  train "$V" --epochs "$EPM" --lr 5e-4 --merge-last 4 --hist "$H" "${KN[@]}" "${MIXK[@]}" "${BH[@]}"
done
# summary: per variant, best record by val gold acc (zero = its epoch -1 record)
uv run --no-sync python - "$OUT" > "$OUT/summary.txt" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
best_v, best_s = "zero", -1.0
best_hv, best_h = "zero", -1.0  # also playtest the best on unseen-deck gold (bcg-20261004-2223: mix won there)
for v in ("zero", "ft-gold", "ft-mix", "ft-teach", "ft-teach-h", "ft-search", "ft-dw", "ft-all", "ft-t13", "ft-swv", "ft-swd", "ft-r2", "ft-r2t", "mix", "lines-gold", "mix-l", "mix-h32", "mix-h128", "mix-h256", "mix-h512",
          "mix-h32k", "mix-h128k", "mix-h256k", "mix-h512k", "mix-k"):
    m = out / v / "metrics.jsonl"
    if not m.exists():
        continue
    recs = [r for r in map(json.loads, filter(str.strip, m.read_text().splitlines())) if "val" in r]
    if not recs:
        continue
    # the record best.pt holds: the last epoch under --select-on last (mix-20261005-2016 summary showed epoch -1 by gold
    # while best.pt was the final epoch)
    sel = next((r["args"].get("select_on") for r in recs if "args" in r), None)
    r = recs[-1] if sel == "last" else max(recs, key=lambda r: r["val"]["per_kind"].get("gold", -1))
    g = r["val"]["per_kind"].get("gold", -1)
    hk = r.get("holdout", {}).get("per_kind", {})
    print(f"{v:8s} at {r.get('epoch', 'merged')}: val gold {g:.4f} gold_pass {r['val']['per_kind'].get('gold_pass', -1):.4f} "
          f"| holdout gold {hk.get('gold', -1):.4f} all {r.get('holdout', {}).get('acc', -1):.4f} | val all {r['val']['acc']:.4f}")
    hg = hk.get("gold", -1)
    if v != "zero" and g > best_s:
        best_v, best_s = v, g
    if v != "zero" and hg > best_h:
        best_hv, best_h = v, hg
print("playtest", best_v, best_hv if best_hv != best_v else "")
PY
BEST=$(tail -1 "$OUT/summary.txt" | cut -d' ' -f2-)
tick "summary (playtest $BEST)"
# playtest budget (bcg-20261004-2223): 10 decks x 6 hands x best-of-8 took the whole 25 min on combos, so the matches
# (real games vs random/greedy) never ran; now 8 decks x 3 hands x best-of-4 to leave time for the matches
PLAY=($BEST); [ "${PT_ZERO:-1}" = "1" ] && PLAY=(zero "${PLAY[@]}")
[ "${PLAY_ALL:-0}" = "1" ] && PLAY=($VARIANTS)  # every trained variant (M1 report cards)
for v in "${PLAY[@]}"; do
  [ "$v" = zero ] && ck="$BASE" || ck="$OUT/$v/best.pt"
  [ -f "$ck" ] || continue
  if grep -q "train-$v \|ALERT $v:" "$OUT/ALERT" 2>/dev/null; then  # a failed variant is never playtested
    echo "ALERT playtest-$v skipped (training alert)" >> "$OUT/ALERT"; continue
  fi
  # playtest writes to <ckpt>/../../playtest and reads the architecture from metrics.jsonl next to the checkpoint
  d="$OUT/pt-$v/ck"; mkdir -p "$d"; cp "$ck" "$d/best.pt"; cp "$OUT/$v/metrics.jsonl" "$d/"
  VS=(); [ "$v" = zero ] || VS=(--vs-ckpt "$BASE")
  timeout "$PT_CAP" uv run --no-sync python scripts/playtest.py --ckpt "$d/best.pt" --device cuda --threads "$J" \
    --seed 0 "${PT[@]}" "${VS[@]}" > "$OUT/playtest-$v.txt" 2>&1
  grep -h "^ALERT\|Traceback" "$OUT/playtest-$v.txt" | sed "s/^/ALERT playtest-$v: /" >> "$OUT/ALERT"
  rm -f "$d/best.pt"; tick "playtest-$v"
done
kill "$USAGE_PID" 2>/dev/null
echo done > "$OUT/DONE"
