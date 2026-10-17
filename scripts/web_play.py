"""Play the product from a phone browser: serve the W1 WebSocket server.

    uv run python scripts/web_play.py [--bind 127.0.0.1|tailscale|<addr>] [--port 8765]
        [--token <token>] [--opponent-default product] [--device cpu] [--replay-threads 1]
"""

from __future__ import annotations

import argparse
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def resolve_bind(bind: str) -> str:
    if bind == "tailscale":
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True,
                             check=True).stdout
        ip = out.strip().splitlines()[0].strip() if out.strip() else ""
        if not ip:
            sys.exit("ALERT web_play: `tailscale ip -4` returned no address")
        return ip
    return bind


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="127.0.0.1 (default) | tailscale (`tailscale ip -4`) | an address")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=None, help="access token (default: random)")
    ap.add_argument("--opponent-default", default="product",
                    help="product (default) | fast | random")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--replay-threads", type=int, default=1)
    a = ap.parse_args()
    host = resolve_bind(a.bind)
    token = a.token or secrets.token_urlsafe(16)

    import uvicorn

    from ygo_zero.web.server import create_app

    app = create_app(token=token, opponent_default=a.opponent_default, device=a.device,
                     replay_threads=a.replay_threads)
    print(f"ygo-zero web client: http://{host}:{a.port}/?t={token}", flush=True)
    uvicorn.run(app, host=host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
