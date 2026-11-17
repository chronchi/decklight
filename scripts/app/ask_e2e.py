"""Ask ws protocol end to end against the real Rust server (docs/plans/ask-tab.md step 5).

Uses this machine's saved OpenCode Go key (`ask_key status`) and makes two short model calls: one answered turn, one
stopped right after it starts. Chats go to a scratch folder, never the app's own.

    uv run python scripts/app/ask_e2e.py [--rust-bin rust/target/release/decklight-serve]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_gate import start_server  # noqa: E402


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", flush=True)
    raise SystemExit(1)


async def until(ws, pred, what: str, timeout: float = 180.0) -> dict:
    async def go():
        while True:
            m = json.loads(await ws.recv())
            if pred(m):
                return m

    try:
        return await asyncio.wait_for(go(), timeout)
    except TimeoutError:
        fail(f"no {what} in {timeout:.0f} s")


async def ask(ws, msg: dict, t: str, req: int) -> dict:
    await ws.send(json.dumps({**msg, "req": req}))
    m = await until(ws, lambda m: m.get("req") == req and m.get("t") in (t, "ask_error"), t, 30)
    if m["t"] == "ask_error":
        fail(f"{msg['t']}: {m}")
    return m


async def run(port: int, token: str) -> None:
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws?t={token}", max_size=None) as ws:
        s = await ask(ws, {"t": "ask_settings"}, "ask_settings", 1)
        go = next(p for p in s["presets"] if p["id"] == "opencode-go")
        print(f"settings: {s['preset']} / {s['model']}, key store {s['key_store']['kind']}, go key {'set' if go['key'] else 'none'}", flush=True)
        if any("key" in p and p["key"] and not p["key"].startswith("••••") for p in s["presets"]):
            fail("a key left the server as more than a hint")
        if not go["key"]:
            fail("no OpenCode Go key saved: run ask_key set opencode-go")
        bad = {"t": "ask_settings", "op": "set", "preset": "custom", "model": "m", "base_url": "ftp://x", "req": 2}
        await ws.send(json.dumps(bad))
        m = await until(ws, lambda m: m.get("req") == 2, "settings error", 30)
        if m["t"] != "ask_error" or "base URL" not in m["message"]:
            fail(f"a bad base URL was accepted: {m}")

        started = await ask(ws, {"t": "ask_send", "text": "¿Qué hace Ash Blossom & Joyous Spring? Una frase."}, "ask_started", 3)
        chat = started["chat"]
        kinds: list[str] = []
        end = await until(ws, lambda m: m.get("t") == "ask_event" and (kinds.append(m["kind"]) or m["kind"] in ("done", "error")), "answer")
        print(f"turn 1 events: {sorted(set(kinds))}; end {end['kind']}", flush=True)
        if end["kind"] != "done" or not end["answer"].strip():
            fail(f"turn 1 did not answer: {end}")
        if "text" not in kinds:
            fail("no streamed text before done")
        print(f"answer: {end['answer'][:160]!r}", flush=True)

        await ws.send(json.dumps({"t": "ask_send", "chat": chat, "text": "¿Y Maxx \"C\"?", "req": 4}))
        busy = {"t": "ask_send", "chat": chat, "text": "otra", "req": 5}
        await until(ws, lambda m: m.get("req") == 4 and m.get("t") == "ask_started", "second start", 30)
        await ws.send(json.dumps(busy))
        m = await until(ws, lambda m: m.get("req") == 5, "busy reply", 30)
        if m.get("action") != "cancel":
            fail(f"a second turn was let in while one runs: {m}")
        await ws.send(json.dumps({"t": "ask_cancel"}))
        end = await until(ws, lambda m: m.get("t") == "ask_event" and m["kind"] in ("done", "error"), "stopped turn")
        print(f"turn 2 after cancel: {end['kind']} {end.get('message', '')!r}", flush=True)
        if end["kind"] != "error" or end["message"] != "Stopped.":
            fail(f"cancel did not stop the turn: {end}")

        chats = await ask(ws, {"t": "ask_list"}, "ask_chats", 6)
        if [c["id"] for c in chats["chats"]] != [chat] or chats["running"] is not None:
            fail(f"chat list wrong: {chats}")
        opened = await ask(ws, {"t": "ask_open", "chat": chat}, "ask_chat", 7)
        roles = [r["role"] for r in opened["rows"]]
        print(f"stored rows: {roles}; tokens {opened['chat']['tokens']}", flush=True)
        if roles.count("user") != 2 or "error" not in roles or opened["chat"]["tokens"] <= 0:
            fail("stored chat is missing the questions, the stop row or the usage")
        first_user = next(r["id"] for r in opened["rows"] if r["role"] == "user")
        forked = await ask(ws, {"t": "ask_fork", "chat": chat, "message": first_user}, "ask_chat", 8)
        if forked["chat"]["head"] != first_user:
            fail("fork did not move the head")
        gone = await ask(ws, {"t": "ask_delete", "chat": chat}, "ask_chats", 9)
        if gone["chats"]:
            fail("delete left the chat")
        print("all checks passed", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rust-bin", type=Path, default=ROOT / "rust/target/release/decklight-serve")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "data/cache/ask_e2e", help="server log, records, chats")
    a = ap.parse_args()
    ask_dir = a.run_dir / "ask"
    shutil.rmtree(ask_dir, ignore_errors=True)
    import room_gate

    orig = room_gate.subprocess.Popen

    def popen(cmd, **kw):  # the scratch chat folder, not the app's
        return orig([*cmd, "--ask-dir", str(ask_dir)], **kw)

    room_gate.subprocess.Popen = popen
    proc, port, token = start_server(a.rust_bin, a.run_dir, "ask")
    try:
        asyncio.run(run(port, token))
    finally:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
