"""A5 end-to-end check: build a card pack, serve it over local HTTP, let the Rust server pick it up,
and start a duel with the new card.

Run with `uv run python scripts/app/pack_e2e.py` (needs `rust/target/release/decklight-serve`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[2]
CODE = 999999001
NAME = "Pack Test Dragon"


def make_source(dir: Path) -> Path:
    src = dir / "source"
    (src / "script").mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(src / "pack-test.cdb")
    con.execute(
        "CREATE TABLE datas (id INT, ot INT, alias INT, setcode INT, type INT, atk INT, def INT, "
        "level INT, race INT, attribute INT, category INT)"
    )
    con.execute(
        "CREATE TABLE texts (id INT, name TEXT, desc TEXT, "
        + ", ".join(f"str{i} TEXT" for i in range(1, 17))
        + ")"
    )
    con.execute("INSERT INTO datas VALUES (?, 1, 0, 0, 1, 1800, 1200, 4, 1, 1, 0)", (CODE,))
    con.execute(
        "INSERT INTO texts VALUES (" + ", ".join("?" * 19) + ")",
        (CODE, NAME, "An effect monster used by the A5 end-to-end check.", *[""] * 16),
    )
    con.commit()
    con.close()
    (src / "script" / f"c{CODE}.lua").write_text(
        "local s,id=GetID()\n"
        "function s.initial_effect(c)\n"
        "\tlocal e1=Effect.CreateEffect(c)\n"
        "\te1:SetType(EFFECT_TYPE_ACTIVATE)\n"
        "\te1:SetCode(EVENT_FREE_CHAIN)\n"
        "\tc:RegisterEffect(e1)\n"
        "end\n"
    )
    return src


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def serve_dir(dir: Path, port: int) -> ThreadingHTTPServer:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(dir), **kw)

        def log_message(self, *_a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def build_pack(work: Path) -> Path:
    src = make_source(work)
    zip_path = work / "pack-test.zip"
    cmd = [
        sys.executable, str(ROOT / "scripts/app/build_card_pack.py"),
        "--source-dir", str(src), "--codes", str(CODE), "--id", "pack-e2e",
        "--fake-emb", "--verify", "--verify-games", "1", "--out", str(zip_path),
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ALERT pack_e2e: build failed\n{r.stdout}\n{r.stderr}")
    return zip_path


def start_server(work: Path, packs: Path, url: str, token: str) -> tuple[subprocess.Popen, int]:
    bin_path = ROOT / "rust/target/release/decklight-serve"
    if not bin_path.is_file():
        raise SystemExit("ALERT pack_e2e: build rust/target/release/decklight-serve first")
    port = free_port()
    env = dict(os.environ)
    env.pop("YGO_EMB", None)
    env.pop("YGO_EXTRA_CARDS", None)
    p = subprocess.Popen(
        [str(bin_path), "--root", str(ROOT), "--port", str(port), "--token", token,
         "--packs-dir", str(packs), "--pack-url", url, "--records", str(work / "records")],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        line = p.stdout.readline() if p.stdout else ""
        if line.startswith("listening on"):
            return p, int(line.split()[-1])
        if p.poll() is not None:
            err = p.stderr.read() if p.stderr else ""
            raise SystemExit(f"ALERT pack_e2e: server exited ({p.returncode})\n{err}")
    p.kill()
    raise SystemExit("ALERT pack_e2e: server did not listen in 60 s")


async def duel(port: int, token: str) -> dict:
    deck = "#main\n" + "\n".join([str(CODE)] * 40) + "\n#extra\n!side\n"
    uri = f"ws://127.0.0.1:{port}/ws?t={token}"
    async with websockets.connect(uri, open_timeout=10) as ws:
        await ws.send(json.dumps({
            "t": "new_game",
            "deck": {"ydk": deck},
            "bot_deck": {"ydk": deck},
            "mode": "random",
            "first": "me",
        }))
        first_state = None
        for _ in range(40):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            if msg.get("t") == "error":
                raise SystemExit(f"ALERT pack_e2e: duel refused: {msg.get('text')}")
            if msg.get("t") == "state" and msg.get("view"):
                first_state = msg
                break
        if first_state is None:
            raise SystemExit("ALERT pack_e2e: no duel state arrived")
        await ws.send(json.dumps({"t": "surrender"}))
        return first_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the scratch dir")
    a = ap.parse_args()
    work = ROOT / "data/cache/app/a5" / f"e2e-{time.strftime('%Y%m%d-%H%M%S')}"
    work.mkdir(parents=True, exist_ok=True)
    token = "pack-e2e"
    try:
        zip_path = build_pack(work)
        port = free_port()
        srv = serve_dir(zip_path.parent, port)
        server, sport = start_server(work, work / "packs", f"http://127.0.0.1:{port}/{zip_path.name}", token)
        try:
            base = f"http://127.0.0.1:{sport}"
            with httpx.Client(timeout=20) as c:
                st = c.get(f"{base}/packs/status?t={token}").json()
                pack = st.get("packs", {})
                if pack.get("state") != "active" or pack.get("id") != "pack-e2e":
                    raise SystemExit(f"ALERT pack_e2e: pack not active: {st}")
                card = c.get(f"{base}/card/{CODE}?t={token}").json()
                if card.get("name") != NAME:
                    raise SystemExit(f"ALERT pack_e2e: card text not loaded: {card}")
            state = asyncio.run(duel(sport, token))
            if not state.get("view"):
                raise SystemExit("ALERT pack_e2e: duel had no view")
            print(json.dumps({
                "ok": True,
                "pack": {"id": pack.get("id"), "cards": pack.get("cards"), "scripts": pack.get("scripts"),
                         "embeddings": pack.get("embeddings")},
                "card": card.get("name"),
                "duel": {"turn": state["view"].get("turn"), "phase": state["view"].get("phase")},
                "zip": str(zip_path),
            }))
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
            srv.shutdown()
    finally:
        if not a.keep:
            import shutil
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
