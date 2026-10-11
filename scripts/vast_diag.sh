#!/usr/bin/env bash
# Encoding diagnostics on a vast box (scripts/diag/encoding_diag.py): collect states from the final card's games on all
# cores, then the policy/value probes on the GPU and the card-text timing probes on the CPU in parallel.
# Env: POLICY, VALUE. Usage: bash scripts/vast_diag.sh <name>. Results: runs/<name>/diag/{probe,text}.json, *.txt, ALERT
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NAME="${1:?usage: bash scripts/vast_diag.sh <name>}"; OUT="runs/$NAME"; D="$OUT/diag"; mkdir -p "$D"
: "${POLICY:?set POLICY}"; : "${VALUE:?set VALUE}"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
uv sync --group dev --group ml -q --no-install-package torch
export UV_HTTP_TIMEOUT=300
for try in 1 2 3; do
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/torch.txt" 2>&1 && break
  echo "torch install attempt $try failed" >> "$OUT/torch.txt"; sleep 10
done
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || { alert "no CUDA torch"; exit 3; }
uv run --no-sync python -c "import sklearn" || { alert "no sklearn"; exit 3; }
read -r Q P < /sys/fs/cgroup/cpu.max 2>/dev/null || Q=max
J=$(nproc); [ "$Q" != max ] && J=$(( Q / P ))
echo "cores $J" > "$OUT/sizing.txt"
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem/{print $3}')G"; sleep 30; done ) > "$OUT/usage.txt" 2>&1 &
USAGE_PID=$!
# text probes need no states: start them now on 4 cores
OMP_NUM_THREADS=4 timeout 60m uv run --no-sync python scripts/diag/encoding_diag.py --stage text --out "$D" > "$OUT/text.txt" 2>&1 &
TEXT_PID=$!
N=$(( J > 4 ? J - 4 : 1 )); [ $N -gt 64 ] && N=64
for p in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=1 timeout 40m uv run --no-sync python scripts/diag/encoding_diag.py --stage collect --out "$D" \
    --games data/cache/diag/games/raw data/cache/diag/games/hppo --nproc $N --proc $p > "$OUT/collect-$p.txt" 2>&1 &
done
wait $(jobs -p | grep -v -e "^$USAGE_PID\$" -e "^$TEXT_PID\$") 2>/dev/null
NS=$(ls "$D"/states-*.npz 2>/dev/null | wc -l)
[ "$NS" -lt "$N" ] && alert "collect: only $NS of $N state files ($(grep -h ALERT $OUT/collect-*.txt | head -2 | tr '\n' ' '))"
echo "PROGRESS collect done: $NS files, $(cat $OUT/collect-*.txt | awk '{s+=$5; w+=$7} END {print s" states, "w" windows"}')" | tee -a "$OUT/PROGRESS"
RC=0
if [ "$NS" -gt 0 ]; then
  timeout 60m uv run --no-sync python scripts/diag/encoding_diag.py --stage probe --out "$D" --policy "$POLICY" --value "$VALUE" \
    > "$OUT/probe.txt" 2>&1 || { RC=$?; alert "probe exit $RC: $(tail -3 $OUT/probe.txt | tr '\n' ' ')"; }
  echo "PROGRESS probe done rc=$RC $(date +%T)" >> "$OUT/PROGRESS"
fi
wait $TEXT_PID || { alert "text exit $?: $(tail -3 $OUT/text.txt | tr '\n' ' ')"; RC=1; }
echo "PROGRESS text done $(date +%T)" >> "$OUT/PROGRESS"
grep -h "^ALERT" "$OUT"/probe.txt "$OUT"/text.txt 2>/dev/null >> "$OUT/ALERT"
rm -f "$D"/states-*.npz   # large, regenerable from the shipped games
kill $USAGE_PID 2>/dev/null
exit $RC
