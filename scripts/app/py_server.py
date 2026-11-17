"""Python oracle server for the A2c transcript gate (uvicorn, fixed seed, StreamBot opponent).

    uv run python scripts/app/py_server.py --port 0 --token T --seed 21000 --records DIR --pics DIR

Prints `listening on <port>` once the server accepts connections, then serves until killed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--token", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--pics", type=Path, required=True)
    a = ap.parse_args(argv)

    import uvicorn

    from decklight.edopro.bot import StreamBot
    from decklight.engine.duel import Engine
    from decklight.web.server import create_app

    app = create_app(
        token=a.token,
        engine=Engine(),
        make_opponent=lambda mode: StreamBot(a.seed),
        record_root=a.records,
        pics_dir=a.pics,
        fixed_seed=a.seed,
    )

    async def run() -> None:
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=a.port, log_level="warning"))
        task = asyncio.create_task(server.serve())
        while not server.started and not task.done():
            await asyncio.sleep(0.02)
        if task.done():
            return
        port = server.servers[0].sockets[0].getsockname()[1]
        print(f"listening on {port}", flush=True)
        await task

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
