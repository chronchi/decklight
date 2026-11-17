"""On-device Gemma in the installed app, driven over adb (no taps): the app's server downloads Gemma 4 E4B through
the device_* messages, then answers two questions with Ask's tools on the phone's GPU. The user's Ask settings are put
back and the app is force-stopped at the end whatever happens, so no engine stays loaded (AGENTS.md "The user's phone
is theirs"). The model stays installed unless --remove.

    uv run --no-project --with websockets==15.0.1 python -u scripts/phone/device_gemma_e2e.py <serial> [--remove]
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time

import websockets

ADB = [str(__import__("pathlib").Path.home() / "Android/Sdk/platform-tools/adb"), "-s", sys.argv[1]]
PKG = "com.decklight.app.debug"
QUESTIONS = ["What does Ash Blossom & Joyous Spring do? One sentence.", "Is Maxx \"C\" a monster or a spell? One line."]
problems: list[str] = []


def adb(*a: str) -> str:
    return subprocess.run(ADB + list(a), capture_output=True, text=True, timeout=120).stdout


async def until(ws, ok, what: str, timeout: float):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        m = json.loads(await asyncio.wait_for(ws.recv(), timeout=max(1, end - time.monotonic())))
        if ok(m):
            return m
    raise TimeoutError(f"no {what} in {timeout:.0f} s")


async def run(port: int, token: str) -> None:
    # The port opens before the server has loaded (HTTP 503 until "server ready", ~5 s on the phone).
    for _ in range(30):
        try:
            ws = await websockets.connect(f"ws://127.0.0.1:{port}/ws?t={token}", max_size=None, ping_interval=None)
            break
        except websockets.exceptions.InvalidStatus:
            await asyncio.sleep(2)
    else:
        raise TimeoutError("the app's server answered 503 for 60 s")
    async with ws:
        await ws.send(json.dumps({"t": "ask_settings", "req": 1}))
        before = await until(ws, lambda m: m.get("req") == 1, "settings", 30)
        print(f"settings before: {before['preset']} / {before['model']}", flush=True)
        try:
            await ws.send(json.dumps({"t": "device_status", "req": 2}))
            st = await until(ws, lambda m: m.get("t") == "device_status", "device_status", 30)
            print("device check:", json.dumps(st["device"]), flush=True)
            if not st["device"]["gpu"]:
                problems.append("the phone check found no OpenCL")
                return
            m0 = st["models"][0]
            if m0["state"] != "installed":
                t0 = time.monotonic()
                await ws.send(json.dumps({"t": "device_download", "model": m0["id"], "req": 3}))
                last = -10
                while True:
                    m = await until(ws, lambda m: m.get("t") in ("device_status", "error"), "download progress", 300)
                    if m["t"] == "error":
                        problems.append(f"download refused: {m}")
                        return
                    g = m["models"][0]
                    pct = int(100 * g["bytes"] / g["total"])
                    if pct >= last + 10 or g["state"] != "downloading":
                        last = pct
                        print(f"download {g['state']} {pct}% {time.monotonic() - t0:.0f} s", flush=True)
                    if g.get("error"):
                        problems.append(f"download failed: {g['error']}")
                        return
                    if g["state"] == "installed":
                        break
                    if time.monotonic() - t0 > 1800:
                        problems.append("download over 30 min")
                        return
            await ws.send(json.dumps({"t": "ask_settings", "op": "set", "preset": "device", "model": m0["id"], "req": 4}))
            await until(ws, lambda m: m.get("req") == 4, "settings set", 30)
            chat = None
            for i, q in enumerate(QUESTIONS):
                t0 = time.monotonic()
                msg = {"t": "ask_send", "text": q, "req": 10 + i}
                if chat:
                    msg["chat"] = chat
                await ws.send(json.dumps(msg))
                started = await until(ws, lambda m: m.get("req") == 10 + i, "ask_started", 30)
                if started["t"] != "ask_started":
                    problems.append(f"send refused: {started}")
                    return
                chat = started["chat"]
                kinds, tools, first = [], [], None
                def seen(m):
                    nonlocal first
                    if m.get("t") != "ask_event":
                        return False
                    kinds.append(m["kind"])
                    if m["kind"] == "text" and first is None:
                        first = time.monotonic() - t0
                    if m["kind"] == "tool_start":
                        tools.append(m.get("name"))
                    return m["kind"] in ("done", "error")
                end = await until(ws, seen, "answer", 300)
                took = time.monotonic() - t0
                print(f"Q{i + 1} {end['kind']} in {took:.1f} s (first text {first and round(first, 1)} s); tools {tools}", flush=True)
                print(f"   {(end.get('answer') or end.get('message') or '')[:300]!r}", flush=True)
                if end["kind"] != "done":
                    problems.append(f"Q{i + 1}: {end}")
            if chat:
                await ws.send(json.dumps({"t": "ask_delete", "chat": chat, "req": 30}))
        finally:
            await ws.send(json.dumps({"t": "ask_settings", "op": "set", "preset": before["preset"], "model": before["model"], "req": 31}))
            await until(ws, lambda m: m.get("req") == 31, "settings restored", 30)
            print("settings restored", flush=True)
            if "--remove" in sys.argv:
                await ws.send(json.dumps({"t": "device_remove", "model": "gemma-4-e4b", "req": 32}))


def main() -> int:
    # A run.json left by an earlier start names a dead port: start the app fresh and wait for its own.
    if not adb("shell", f"pidof {PKG}").strip():
        adb("shell", f"run-as {PKG} rm -f root/logs/run.json")
        adb("shell", f"am start -n {PKG}/com.decklight.app.MainActivity")
    run_json = ""
    for _ in range(30):
        run_json = adb("shell", f"run-as {PKG} cat root/logs/run.json 2>/dev/null")
        if run_json.strip():
            break
        time.sleep(2)
    if not run_json.strip():
        print("ALERT: the app wrote no run.json in 60 s", flush=True)
        adb("shell", f"am force-stop {PKG}")
        return 1
    r = json.loads(run_json)
    # One connection at a time and the newest wins (server.rs takeover, 4400); the app's own screen does not take it
    # back after a 4400 (web/ws.js). So let the screen connect first, then take over.
    time.sleep(20)
    subprocess.run(ADB + ["forward", f"tcp:{r['port']}", f"tcp:{r['port']}"], check=True)
    try:
        asyncio.run(run(r["port"], r["token"]))
    except Exception as e:  # noqa: BLE001 — reported, then the cleanup below still runs
        problems.append(f"{type(e).__name__}: {e}")
    finally:
        subprocess.run(ADB + ["forward", "--remove", f"tcp:{r['port']}"])
        adb("shell", f"am force-stop {PKG}")
        left = adb("shell", f"pidof {PKG}").strip()
        print("app stopped" if not left else f"ALERT: app still running ({left})", flush=True)
    print("problems", problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
