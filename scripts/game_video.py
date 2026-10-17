"""Render a recorded match to MP4 with the web client (spectator, both hands visible).

    uv run python scripts/game_video.py runs/<job>/<variant>/games-0.jsonl --idx 0 \\
        --deck0 decks/pool/snake-eye.ydk --deck1 decks/custom/wiccanthrope.ydk \\
        --name0 "Snake-Eye" --name1 "Wiccanthrope" --out data/cache/video/g2.mp4

The input is a `search_match` games-*.jsonl record (the same input
`scripts/export_yrp.py` takes). Frames come from `src/ygo_zero/web/replay.py`
and are served to web/app.js `?replay=` mode by the real Starlette app; each
frame is screenshotted with Playwright Chromium and encoded to H.264 with
imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import imageio_ffmpeg
import uvicorn
from starlette.responses import JSONResponse
from starlette.routing import Route
from web_screens import launch_kwargs


def deck_path(name: str) -> Path:
    p = Path(name)  # an explicit path loads directly
    if p.is_file():
        return p
    if (ROOT / name).is_file():
        return ROOT / name
    stem = Path(name).stem
    for cand in (
        ROOT / "decks" / f"{name}.ydk",
        ROOT / "decks" / f"{name.lstrip('/')}.ydk",
        ROOT / "decks" / f"{stem}.ydk",
    ):
        if cand.is_file():
            return cand
    for d in ("pool", "gauntlet", "newcards", "custom"):
        for cand in ((ROOT / "decks" / d / f"{name}.ydk"),
                     (ROOT / "decks" / d / f"{stem}.ydk")):
            if cand.is_file():
                return cand
    sys.exit(f"deck {name} not found (tried explicit path and decks/pool|gauntlet|newcards|custom)")


def hold_for(frame: dict, prev: dict | None, base: float) -> float:
    """Per-frame hold: base, x2 on chain/activation or a phase/turn change."""
    h = base
    kinds = {e.get("kind") for e in frame.get("log", [])}
    text = " ".join(e.get("text", "") for e in frame.get("log", [])).lower()
    if "activate" in kinds or "chain" in text:
        return h * 2
    v, pv = frame.get("view", {}), (prev or {}).get("view", {})
    if prev is not None and (v.get("turn") != pv.get("turn")
                             or v.get("phase") != pv.get("phase")):
        return h * 2
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", type=Path, help="a search_match games-*.jsonl")
    ap.add_argument("--idx", type=int, required=True, help="game_idx of the record")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name0", default="Player 1")
    ap.add_argument("--name1", default="Player 2")
    ap.add_argument("--deck0", type=Path, default=None)
    ap.add_argument("--deck1", type=Path, default=None)
    ap.add_argument("--size", default="720x1280", help="WxH viewport (default phone portrait)")
    ap.add_argument("--hold", type=float, default=1.1, help="base seconds per frame")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None, help="render only the first N frames")
    a = ap.parse_args()

    rec = next(
        (json.loads(x) for x in a.games.read_text().splitlines()
         if json.loads(x)["game_idx"] == a.idx), None
    )
    if rec is None:
        sys.exit(f"no game_idx {a.idx} in {a.games}")
    if rec.get("truncated"):
        print(f"ALERT game {a.idx}: record is truncated", file=sys.stderr)
        sys.exit(2)
    d0, d1 = rec["decks"]
    from ygo_zero.decks import load_ydk
    from ygo_zero.engine.duel import DuelConfig, Engine
    from ygo_zero.web.replay import replay_frames

    cfg = DuelConfig(load_ydk(a.deck0 or deck_path(d0)),
                     load_ydk(a.deck1 or deck_path(d1)), seed=rec["seed"])
    engine = Engine()
    try:
        frames = replay_frames(
            engine, cfg, rec["choices"], (a.name0, a.name1),
            expect_winner=rec["winner"],
            auto_single=rec.get("auto_single", True),
            surrendered=rec.get("surrendered", False))
    except RuntimeError as e:
        print(f"ALERT game {a.idx}: {e}", file=sys.stderr)
        sys.exit(2)
    assert len(frames) == len(rec["choices"]) + 1, (len(frames), len(rec["choices"]))

    if a.max_frames is not None:
        frames = frames[: a.max_frames]
    w, h = (int(x) for x in a.size.lower().split("x"))

    from ygo_zero.web.server import create_app

    token = secrets.token_hex(8)
    app = create_app(token=token)
    payload = {"frames": frames}

    async def replay_json(req):
        if req.query_params.get("t") != token:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return JSONResponse(payload)

    app.routes.insert(len(app.routes) - 1, Route("/replay/video.json", replay_json))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/?replay=/replay/video.json&t={token}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/?t={token}", timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    else:
        sys.exit("ALERT game_video: server did not start")

    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    durations: list[float] = []
    prev: dict | None = None
    for i, f in enumerate(frames):
        if i == len(frames) - 1 and not (a.max_frames is not None
                                          and len(frames) < len(rec["choices"]) + 1):
            durations.append(4.0)  # the final board lingers
        elif i == 0:
            durations.append(max(a.hold, 2.5))  # the title card
        else:
            durations.append(hold_for(f, prev, a.hold))
        prev = f

    tmp = Path(tempfile.mkdtemp(prefix="game-video-"))
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs())
            page = browser.new_page(viewport={"width": w, "height": h},
                                    device_scale_factor=1)
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console-{m.type}: {m.text}")
                    if m.type == "error" else None)
            page.goto(url, wait_until="networkidle")
            page.wait_for_function("window.frameCount > 0", timeout=30000)
            n = page.evaluate("window.frameCount")
            assert n == len(frames), f"client loaded {n} frames, have {len(frames)}"
            last_hash, run = None, 0
            for i in range(len(frames)):
                page.evaluate(f"showFrame({i})")
                png = tmp / f"frame-{i:04d}.png"
                page.screenshot(path=str(png))
                digest = hashlib.sha256(png.read_bytes()).hexdigest()
                run = run + 1 if digest == last_hash else 0
                last_hash = digest
                if run > 30:
                    print(f"ALERT game_video: {run} byte-identical frames in a row "
                          f"(rendering stuck at frame {i})", file=sys.stderr)
                    sys.exit(1)
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    if errors:
        for e in errors[:5]:
            print(f"ALERT game_video: {e}", file=sys.stderr)
        sys.exit(1)

    exe = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    lst = tmp / "frames.ffconcat"
    with lst.open("w") as f:
        f.write("ffconcat version 1.0\n")
        for i, d in enumerate(durations):
            f.write(f"file '{tmp / f'frame-{i:04d}.png'}'\nduration {d:.3f}\n")
        f.write(f"file '{tmp / f'frame-{len(frames) - 1:04d}.png'}'\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(a.fps),
           "-movflags", "+faststart", str(a.out)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print(f"ALERT game_video: ffmpeg failed:\n{r.stderr[-2000:]}", file=sys.stderr)
        sys.exit(1)
    size = a.out.stat().st_size
    if size < 10 * 1024:
        print(f"ALERT game_video: mp4 too small ({size} bytes)", file=sys.stderr)
        sys.exit(1)
    for tag, i in (("first", 0), ("mid", len(frames) // 2), ("last", len(frames) - 1)):
        dst = a.out.parent / f"{a.out.stem}-{tag}.png"
        shutil.copyfile(tmp / f"frame-{i:04d}.png", dst)
    lines = []
    for i, (f, d) in enumerate(zip(frames, durations)):
        c = f.get("caption") or {}
        cap = (c.get("actor") + " — " + c.get("text")) if c.get("text") else ""
        lines.append(f"{i} {d:.3f} {' '.join(cap.split())}\n")
    Path(str(a.out) + ".frames.txt").write_text("".join(lines))
    total = sum(durations)
    print(f"frames={len(frames)} duration={total:.1f}s size={size}B out={a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
