#!/usr/bin/env bash
# RL v0 (PLAN Phase 5 first feel, 2026-10-05): PPO self-play from the BC checkpoint in data/cache/rl_init/
# (best.pt + metrics.jsonl), KL-anchored to it; gauntlet eval (vs BC, vs ygo-agent) at iteration 0, every
# EVAL_EVERY iterations and at the end. Usage: bash scripts/vast_rl.sh <name> [smoke]. Env: RL_HOURS (default 1.5),
# RL_ARGS (extra train_rl.py args). Results in runs/<name>/ (pulled back): rl/metrics.jsonl + iter-*.pt, train.txt.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="$1"; MODE="${2:-full}"; OUT="runs/$NAME"; mkdir -p "$OUT"
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
if [ "$Q" = "max" ]; then J=$(nproc); else J=$(( Q / P )); fi
[ "$J" -lt 2 ] && J=2
T0=$(date +%s)
uv sync --group dev --group ml --group gauntlet -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
echo "deps install $(( $(date +%s) - T0 ))s" >> "$OUT/torch.txt"
{ nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; echo "jobs $J"; free -g | head -2;
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; } > "$OUT/machine.txt" 2>&1
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA torch: abort" | tee -a "$OUT/torch.txt"; exit 3; }
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -m | awk '/Mem/{print $3}')"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
INIT=data/cache/rl_init/best.pt
rl() {  # <out> <args...>
  local o="$1"; shift
  # shellcheck disable=SC2086
  uv run --no-sync python scripts/train_rl.py --init "$INIT" --out "$o" --device cuda --threads "$J" "$@" ${RL_ARGS:-}
}
# smoke first (a few small iterations + evals): a crash here stops the job before the long run
rl "$OUT/smoke" --iters 3 --rollout 4096 --envs 64 --eval-every 2 --eval-seeds 1 --ga-seeds 1 --value-warmup 1 \
  > "$OUT/smoke.txt" 2>&1
echo "exit $?" >> "$OUT/smoke.txt"
if [ "$MODE" != "smoke" ] && grep -q "^done:" "$OUT/smoke.txt"; then
  rl "$OUT/rl" --hours "${RL_HOURS:-1.5}" --eval-every "${EVAL_EVERY:-10}" > "$OUT/train.txt" 2>&1
  echo "exit $?" >> "$OUT/train.txt"
fi
rm -rf "$OUT/smoke"/*.pt
# keep the eval checkpoints small: last.pt (with optimizer) only for a resume; drop it to save the pull
rm -f "$OUT/rl/last.pt"
kill "$USAGE_PID" 2>/dev/null
echo done > "$OUT/DONE"
