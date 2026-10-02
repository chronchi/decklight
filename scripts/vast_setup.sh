#!/usr/bin/env bash
# Instance setup (run by the launcher in /workspace/ygo-zero): uv, Python deps, core + C++ env.
set -euo pipefail
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; }
export PATH="$HOME/.local/bin:$PATH"
command -v g++ >/dev/null || { apt-get update -qq && apt-get install -y -qq g++ >/dev/null; }
echo "cpus $(nproc), $(free -g | awk '/Mem/{print $2}') GiB RAM, $(g++ --version | head -1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no GPU"
uv sync --group dev -q
scripts/build_core.sh > /dev/null
scripts/build_ext.sh > /dev/null
echo "setup done"
