"""End-to-end check of the background card-art sync (never contacts ygoprodeck.com).

Starts `decklight-serve` against the repo's assets with a temp pics dir, a local fake image
server, and a restricted target (`DECKLIGHT_ART_CODES_LIMIT=300`). Drives the real WebSocket
protocol through a tiny stdlib client and checks:

  cellular -> waiting_wifi; allow data -> done; all target files saved exactly once;
  restart -> 0 new fetches; deleting 10 files + restart -> exactly 10;
  a 403 -> rate_limited.

Run: uv run python scripts/app/art_sync_e2e.py [--bin PATH]
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import os
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[2]
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 4096


class FakeImages(http.server.BaseHTTPRequestHandler):
    counts: ClassVar[dict[int, int]] = {}
    forbidden: ClassVar[set[int]] = set()
    notfound_every: ClassVar[int] = 37

    def log_message(self, *args):
        pass

    def do_GET(self):
        m = re.fullmatch(r"/images/cards_small/(\d+)\.jpg", self.path)
        if not m:
            self.send_response(404)
            self.end_headers()
            return
        code = int(m.group(1))
        type(self).counts[code] = type(self).counts.get(code, 0) + 1
        if code in type(self).forbidden:
            self.send_response(403)
            self.end_headers()
            return
        if code % type(self).notfound_every == 0:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(JPEG)))
        self.end_headers()
        self.wfile.write(JPEG)


def start_images(forbidden: set[int] = frozenset()) -> tuple[http.server.ThreadingHTTPServer, int]:
    FakeImages.counts = {}
    FakeImages.forbidden = set(forbidden)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeImages)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


class WS:
    """The small subset of RFC 6455 the e2e needs: one text frame at a time, client masking."""

    def __init__(self, port: int, token: str):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=15)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET /ws?t={token} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            head += self.sock.recv(4096)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"websocket upgrade refused: {head[:120]!r}")

    def _read(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            chunk = self.sock.recv(n - len(out))
            if not chunk:
                raise RuntimeError("server closed the websocket")
            out += chunk
        return out

    def send(self, obj: dict) -> None:
        data = json.dumps(obj).encode()
        mask = os.urandom(4)
        n = len(data)
        header = bytearray([0x81])
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack("!H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", n)
        header += mask
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(bytes(header) + body)

    def recv(self, timeout: float = 2.0) -> dict | None:
        self.sock.settimeout(timeout)
        try:
            b1, b2 = self._read(2)
        except TimeoutError:
            return None
        opcode = b1 & 0x0F
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack("!H", self._read(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", self._read(8))[0]
        payload = self._read(n)
        if opcode == 8:
            raise RuntimeError("server closed the websocket")
        if opcode == 9:  # ping -> pong
            self.sock.sendall(bytes([0x8A, 0x80]) + os.urandom(4))
            return self.recv(timeout)
        if opcode != 1:
            return self.recv(timeout)
        return json.loads(payload)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def wait_art(ws: WS, pred, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        m = ws.recv(timeout=1.0)
        if m is None:
            continue
        if m.get("t") == "art_sync":
            last = m
            if pred(m):
                return m
    raise AssertionError(f"art_sync condition not reached; last frame: {last}")


class Server:
    def __init__(self, root: Path, pics: Path, token: str, img_port: int, records: Path):
        env = os.environ.copy()
        env.update(
            {
                "DECKLIGHT_IMG_BASE": f"http://127.0.0.1:{img_port}",
                "DECKLIGHT_ART_CODES_LIMIT": "300",
                "DECKLIGHT_NO_WARM": "1",
                "RUST_BACKTRACE": "1",
            }
        )
        self.token = token
        self.proc = subprocess.Popen(
            [
                str(BIN),
                "--root",
                str(root),
                "--pics-dir",
                str(pics),
                "--records",
                str(records),
                "--token",
                token,
                "--port",
                "0",
            ],
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.lines: list[str] = []
        port = None
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.proc.stdout], [], [], 1.0)
            if not ready:
                if self.proc.poll() is not None:
                    break
                continue
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    break
                continue
            self.lines.append(line.rstrip())
            m = re.search(r"listening on (\d+)", line)
            if m:
                port = int(m.group(1))
                break
        if port is None:
            self.stop()
            raise RuntimeError("server did not start:\n" + "\n".join(self.lines[-20:]))
        self.port = port

    def ws(self) -> WS:
        return WS(self.port, self.token)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)


def wait_done(ws: WS, timeout: float = 180.0) -> dict:
    return wait_art(ws, lambda m: m["state"] == "done", timeout)


def start_and_run(server: Server) -> tuple[WS, dict]:
    ws = server.ws()
    ws.send({"t": "art_sync", "network": "wifi"})
    ws.send({"t": "art_sync_resume"})
    done = wait_done(ws)
    return ws, done


def main() -> int:
    global BIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(ROOT / "rust/target/release/decklight-serve"))
    args = ap.parse_args()
    BIN = Path(args.bin)
    if not BIN.is_file():
        print(f"skipping: {BIN} is not built (cargo build --release -p decklight-server)", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="decklight-art-e2e-"))
    results: dict[str, object] = {}
    try:
        pics = tmp / "pics"
        records = tmp / "records"
        images_httpd, img_port = start_images()

        # 1. Cellular waits; allowing mobile data downloads everything once.
        srv = Server(ROOT, pics, "e2e-token", img_port, records)
        try:
            ws = srv.ws()
            ws.send({"t": "art_sync", "network": "cellular"})
            ws.send({"t": "art_sync_resume"})
            waiting = wait_art(ws, lambda m: m["state"] == "waiting_wifi", 20)
            assert FakeImages.counts == {}, "cellular fetched before permission"
            ws.send({"t": "art_sync_allow_data"})
            done = wait_done(ws)
            assert done["have"] + done["unavailable"] == done["total"], done
            files = list((pics / "small").glob("*.jpg"))
            assert len(files) == done["have"], (len(files), done)
            assert done["unavailable"] > 0, "the fake 404s were not recorded"
            assert len(FakeImages.counts) == done["have"] + done["unavailable"], FakeImages.counts
            assert done["total"] == 300, done["total"]
            saved = {int(p.stem) for p in files}
            missing = json.loads((pics / "missing.json").read_text())
            assert saved.isdisjoint(int(c) for c in missing), "a missing code was saved"
            results["first_run"] = {
                "total": done["total"],
                "have": done["have"],
                "unavailable": done["unavailable"],
                "requests": len(FakeImages.counts),
                "waiting_state": waiting["state"],
            }
        finally:
            srv.stop()

        # 2. Restart: nothing to fetch.
        before = dict(FakeImages.counts)
        srv = Server(ROOT, pics, "e2e-token", img_port, records)
        try:
            ws, done = start_and_run(srv)
            assert done["state"] == "done"
            extra = {c: n - before.get(c, 0) for c, n in FakeImages.counts.items() if n > before.get(c, 0)}
            assert not extra, f"restart refetched {extra}"
            results["restart"] = {"new_requests": 0, "have": done["have"]}
        finally:
            srv.stop()

        # 3. Delete 10 saved files + restart: exactly those 10 are fetched.
        saved = sorted((pics / "small").glob("*.jpg"))
        deleted = saved[:10]
        for p in deleted:
            p.unlink()
        vanished = {int(p.stem) for p in deleted}
        before = dict(FakeImages.counts)
        srv = Server(ROOT, pics, "e2e-token", img_port, records)
        try:
            ws, done = start_and_run(srv)
            extra = {c: n - before.get(c, 0) for c, n in FakeImages.counts.items() if n > before.get(c, 0)}
            assert set(extra) == vanished, f"expected {len(vanished)} refetches, got {extra}"
            assert all(n == 1 for n in extra.values()), extra
            results["delete_10"] = {"refetched": len(extra)}
        finally:
            srv.stop()

        # 4. A 403 pauses the shared throttle and the sync says so.
        images_httpd.shutdown()
        images_httpd.server_close()
        forbidden_httpd, forbidden_port = start_images()
        pics2 = tmp / "pics403"
        # Forbid every code: the first fetch is a 403 and the shared throttle pauses for an hour.
        FakeImages.forbidden = set(range(10_000_000))
        srv = Server(ROOT, pics2, "e2e-token", forbidden_port, records)
        try:
            ws = srv.ws()
            ws.send({"t": "art_sync", "network": "wifi"})
            ws.send({"t": "art_sync_resume"})
            limited = wait_art(ws, lambda m: m["state"] == "rate_limited", 30)
            assert limited["resume_in_s"] > 600, limited
            results["403"] = {"state": limited["state"], "resume_in_s": limited["resume_in_s"]}
        finally:
            srv.stop()
            forbidden_httpd.shutdown()
            forbidden_httpd.server_close()

        print("art-sync e2e ok")
        print(json.dumps(results, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - the script is the test
        print(f"art-sync e2e FAILED: {exc}", file=sys.stderr)
        for name in ("missing.json", "sync.json"):
            p = tmp / "pics" / name
            if p.exists():
                print(f"{name}: {p.read_text()[:500]}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
