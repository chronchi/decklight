#!/usr/bin/env python3
"""Cancel e2e (user 2026-11-11: "there's no way to cancel the action ... I click set a card ... I can't even cancel"):
a product game (not practice) against the real Rust server. At our first idle prompt with a Set option: Cancel there is
refused ("Nothing to cancel"); pick Set, the follow-up prompt must offer can_cancel; Cancel must bring back the same
idle prompt (same option labels) with can_cancel false; then Set again and finish the placement, and the duel goes on.

    uv run python scripts/app/cancel_e2e.py [--rust-bin rust/target/release/decklight-serve] [--deck decks/pool/mitsurugi.ydk]

Exit 1 on any failed check (ALERT line), 0 when every check passes. Smoke-sized (< 2 min on the laptop).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_gate import start_server  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def fail(msg: str) -> None:
    print(f"ALERT cancel_e2e: {msg}", flush=True)
    raise SystemExit(1)


async def next_state(ws, timeout=180.0) -> dict:
    """The next state frame that waits on us (prompt set, not thinking); errors are returned as-is."""
    while True:
        f = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if f.get("t") == "error":
            return f
        if f.get("t") == "state" and f.get("thinking") is None and (f.get("prompt") or f.get("result")):
            return f


async def run(port: int, token: str, ydk: str) -> None:
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws?t={token}", max_size=None) as ws:
        await ws.send(json.dumps({"t": "new_game", "deck": {"ydk": ydk}, "bot_deck": {"ydk": ydk}, "first": "me",
                                  "chain": "auto", "mode": "product"}))
        for step in range(400):
            f = await next_state(ws)
            if f.get("t") == "error":
                fail(f"server error before the check: {f}")
            if f.get("result"):
                fail("the duel ended before an idle prompt with a Set option")
            p = f["prompt"]
            sets = [o for o in p.get("options", []) if p["kind"] == "idle" and o.get("verb") == "set"]
            if not sets:
                await ws.send(json.dumps({"t": "choose", "prompt": p.get("id"), "index": 0}))
                continue
            labels = [o.get("label") for o in p["options"]]
            print(f"idle prompt with {len(sets)} Set option(s), can_cancel={f.get('can_cancel')}", flush=True)
            if f.get("can_cancel"):
                fail("can_cancel is true at an idle prompt")
            await ws.send(json.dumps({"t": "cancel"}))
            e = await next_state(ws, 30)
            if e.get("t") != "error" or "Nothing to cancel" not in e.get("text", ""):
                fail(f"cancel at the idle prompt was not refused: {e}")
            await ws.send(json.dumps({"t": "choose", "prompt": p.get("id"), "index": sets[0]["index"]}))
            g = await next_state(ws)
            if g.get("t") == "error" or g.get("result"):
                fail(f"after Set: {g}")
            print(f"after Set: prompt {g['prompt']['kind']} can_cancel={g.get('can_cancel')}", flush=True)
            if g["prompt"]["kind"] in ("idle", "battle"):
                fail("Set needed no follow-up prompt (pick a deck whose Set asks for a zone)")
            if not g.get("can_cancel"):
                fail("the follow-up prompt does not offer Cancel")
            await ws.send(json.dumps({"t": "cancel"}))
            h = await next_state(ws)
            if h.get("t") == "error":
                fail(f"cancel refused: {h}")
            back = [o.get("label") for o in h["prompt"].get("options", [])]
            print(f"after Cancel: prompt {h['prompt']['kind']} can_cancel={h.get('can_cancel')}", flush=True)
            if h["prompt"]["kind"] != "idle" or back != labels:
                fail(f"Cancel did not restore the idle prompt: {h['prompt']['kind']} {back[:4]} vs {labels[:4]}")
            if h.get("can_cancel"):
                fail("can_cancel still true after the cancel")
            hs = [o for o in h["prompt"]["options"] if o.get("verb") == "set"]
            await ws.send(json.dumps({"t": "choose", "prompt": h["prompt"].get("id"), "index": hs[0]["index"]}))
            k = await next_state(ws)
            await ws.send(json.dumps({"t": "choose", "prompt": k["prompt"].get("id"), "index": 0}))
            m = await next_state(ws)
            if m.get("t") == "error":
                fail(f"placing after the cancel failed: {m}")
            print(f"placed after Cancel: next prompt {m.get('prompt', {}).get('kind')}; all checks passed", flush=True)
            return
        fail("no idle prompt with a Set option in 400 steps")


def main() -> int:
    import glob
    import os

    if not os.environ.get("ORT_DYLIB_PATH"):  # the product bot loads ONNX Runtime from the venv's onnxruntime wheel
        libs = sorted(glob.glob(str(ROOT / ".venv/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so.*")))
        if libs:
            os.environ["ORT_DYLIB_PATH"] = libs[-1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rust-bin", type=Path, default=ROOT / "rust/target/release/decklight-serve")
    ap.add_argument("--deck", type=Path, default=ROOT / "decks/pool/mitsurugi.ydk")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "data/cache/cancel_e2e", help="server log and records")
    a = ap.parse_args()
    proc, port, token = start_server(a.rust_bin, a.run_dir, "cancel")
    try:
        asyncio.run(run(port, token, a.deck.read_text(encoding="utf-8")))
    finally:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
