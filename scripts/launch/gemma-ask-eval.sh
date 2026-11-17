#!/usr/bin/env bash
# Ask-tab eval of on-device candidates Gemma 4 E2B / E4B (Q4_0 GGUF, the phone's quantization) on a rented GPU
# (user 2026-11-14: "if we want to test the laptop shouldn't we be able to use a vast machine for 10 cents to test it?";
# the laptop CPU took 91-168 s per question). The box only serves llama-server (CUDA image); ask_eval runs on the
# laptop through an ssh tunnel, so tools see the laptop's cards, decks and records exactly as in the Go eval.
# Measures answer quality, not phone speed. Hard cap: a local timer destroys the box after CAP_MIN minutes.
cd "$(dirname "$0")/../.."
set -u
N=gemma-ask-$(date +%Y%m%d-%H%M); OUT=data/cache/briefs/$N.launcher.log; echo "$N -> $OUT"
exec > >(tee -a "$OUT") 2>&1
CAP_MIN=${CAP_MIN:-50}
HF=https://huggingface.co/ggml-org
SETUP="set -e; command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl); mkdir -p /m; cd /m;
for m in E2B E4B; do curl -sSfL -o gemma-4-\$m-it-Q4_0.gguf $HF/gemma-4-\$m-it-GGUF/resolve/main/gemma-4-\$m-it-Q4_0.gguf; done; ls -la /m;
cd /app; export LD_LIBRARY_PATH=/app;  # the image's binary finds its .so files only from /app
setsid nohup ./llama-server -m /m/gemma-4-E2B-it-Q4_0.gguf --jinja -c 32768 -ngl 99 --port 8080 --host 127.0.0.1 > /m/e2b.log 2>&1 < /dev/null &
setsid nohup ./llama-server -m /m/gemma-4-E4B-it-Q4_0.gguf --jinja -c 32768 -ngl 99 --port 8081 --host 127.0.0.1 > /m/e4b.log 2>&1 < /dev/null &
for i in \$(seq 90); do curl -sf 127.0.0.1:8080/health && curl -sf 127.0.0.1:8081/health && break; sleep 2; done; echo;
curl -sf 127.0.0.1:8081/health || { tail -3 /m/e2b.log /m/e4b.log; exit 4; }; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv"
uv run --no-sync python -u - <<EOF || { echo "ALERT session start failed"; exit 3; }
from decklight.infra import vast
spec = vast.JobSpec(name="$N", command="true", hours_cap=$CAP_MIN/60, max_dph=${MAX_DPH:-0.15}, min_cpus=4,
    min_gpu_ram_gb=16, disk_gb=30, image="ghcr.io/ggml-org/llama.cpp:server-cuda", setup='''$SETUP''',
    max_inet_down_cost=${MAX_INET:-0.01}, min_compute_cap=750)
vast.start_session(spec, name="gemma")
EOF
# Hard cap: whatever happens below, the box is destroyed after CAP_MIN minutes.
( sleep $((CAP_MIN * 60)); echo "ALERT cap reached, destroying"; uv run --no-sync python scripts/vast_session.py stop --session gemma ) &
CAP_PID=$!
read -r H P < <(uv run --no-sync python -c "from decklight.infra import vast; s=vast.require_session('gemma'); print(s['host'], s['port'])")
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -N -L 8180:127.0.0.1:8080 -L 8181:127.0.0.1:8081 -p "$P" root@"$H" &
TUN=$!; sleep 5
# One eval at a time: both write the same records fixture sqlite (in parallel: "readonly database", "disk I/O error").
for pair in "E2B 8180" "E4B 8181"; do set -- $pair
  curl -sf 127.0.0.1:$2/health >/dev/null || echo "ALERT $1 server not answering through the tunnel"
  timeout $(((CAP_MIN - 8) * 60 / 2)) taskset -c 0-3 rust/target/release/ask_eval --preset custom --base-url http://127.0.0.1:$2/v1 \
    --model gemma-4-$1-it-Q4_0.gguf > data/cache/briefs/$N.$1.log 2>&1 || echo "ALERT $1 eval exited $?"
done
for m in E2B E4B; do echo "== $m"; grep -E "^\[|ask_eval" data/cache/briefs/$N.$m.log; done
kill $TUN $CAP_PID 2>/dev/null
uv run --no-sync python scripts/vast_session.py stop --session gemma
