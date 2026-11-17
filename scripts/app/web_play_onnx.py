"""Python oracle server for the A3b-2 product gate: the web server's product opponent with ONNX forwarding.

    ORT_DYLIB_PATH=.venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.1 \\
        uv run python scripts/app/web_play_onnx.py --port 0 --token T --seed 7000 \\
            --records DIR --pics DIR --replay-threads 1

`ProductLauncher`'s `base_logits` and `value` are patched to the exact ONNX calls `product_games_py.py` makes
(imported from that script, not copied), `YGO_SEARCH_CLONE=0`, and the C++ core is used (no `YGO_CORE`). `src/` is
untouched. Prints `listening on <port>` once the server accepts connections.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# product_games_py pops YGO_CORE and sets YGO_SEARCH_CLONE=0 at import, and holds the ONNX feeder we reuse.
from scripts.app.product_games_py import OrtNets

_LAUNCHER = None


def _launcher(args):
    global _LAUNCHER
    if _LAUNCHER is not None:
        return _LAUNCHER
    from decklight.edopro.product_bot import ProductLauncher

    launcher = ProductLauncher(replay_threads=args.replay_threads)
    nets = OrtNets(args.onnx, Path(launcher.emb), None)
    launcher.base_logits = nets.policy_logits  # the policy forward on rollout and S outer observations
    launcher.value = nets.values_batch  # the value net on a stacked batch dict (win chance too)
    _LAUNCHER = launcher
    return launcher


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--token", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--pics", type=Path, required=True)
    ap.add_argument("--replay-threads", type=int, default=1)
    ap.add_argument("--onnx", type=Path, default=ROOT / "data/cache/onnx/r1")
    a = ap.parse_args(argv)

    import uvicorn

    from decklight.engine.duel import Engine
    from decklight.web.server import create_app

    launcher = _launcher(a)
    app = create_app(
        token=a.token,
        engine=Engine(),
        make_opponent=lambda mode, model=None: launcher,
        record_root=a.records,
        pics_dir=a.pics,
        fixed_seed=a.seed,
        replay_threads=a.replay_threads,
    )

    async def run() -> None:
        import asyncio

        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=a.port, log_level="warning"))
        task = asyncio.create_task(server.serve())
        while not server.started and not task.done():
            await asyncio.sleep(0.02)
        if task.done():
            return
        port = server.servers[0].sockets[0].getsockname()[1]
        print(f"listening on {port}", flush=True)
        await task

    import asyncio

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
