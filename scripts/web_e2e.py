"""Browser end-to-end check of the web client against the real server.

Starts scripts/web_play.py, opens the page in Playwright Chromium at phone size, starts a duel from the lobby and
plays it by tapping like a person would (glowing cards, menu/sheet options, phase buttons) until the game-over modal.
Saves screenshots to data/cache/web-e2e/. Exits non-zero on a page error, a stall, or no game over.

  uv run --with playwright python scripts/web_e2e.py [--mode random|fast|product] [--max-s 240]
"""
from __future__ import annotations

import argparse
import random
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from web_screens import launch_kwargs  # noqa: E402

OUT = ROOT / "data" / "cache" / "web-e2e"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="random", choices=["random", "fast", "product"])
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--max-s", type=float, default=240)
    ap.add_argument("--shot-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    rng = random.Random(args.seed)
    tok = secrets.token_hex(8)
    url = f"http://127.0.0.1:{args.port}/?t={tok}"
    srv = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "web_play.py"), "--port", str(args.port),
                            "--token", tok], cwd=ROOT, stdout=open(OUT / "server.log", "w"),
                           stderr=subprocess.STDOUT)
    errors: list[str] = []
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        else:
            print("server did not start; see", OUT / "server.log")
            return 1
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs())
            page = browser.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2,
                                    is_mobile=True, has_touch=False)
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e} | {getattr(e, 'stack', '')}"))
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                    if m.type == "error" and "404" not in m.text else None)
            frames: list[str] = []
            page.on("websocket", lambda ws: ws.on("framereceived", lambda f: frames.append(f)))
            page.goto(url)
            page.wait_for_selector("#startbtn", timeout=10000)
            page.screenshot(path=str(OUT / "00-lobby.png"))
            page.click(f"[data-mode={args.mode}]")
            page.click("#startbtn")
            t0, clicks, last_change, last_html, shots = time.time(), 0, time.time(), "", 0
            over = False
            while time.time() - t0 < args.max_s:
                if errors:
                    break
                if page.locator("[data-act=rematch]").count():
                    over = True
                    break
                html = page.inner_html("body")
                if html != last_html:
                    last_html, last_change = html, time.time()
                elif time.time() - last_change > 30:
                    errors.append("stall: page unchanged for 30 s")
                    break
                opts = [o for o in page.locator("[data-opt]:not([disabled])").all() if o.is_visible()]
                acts = [a for a in page.locator(".actionable").all() if a.is_visible()]
                target = None
                if opts and (not acts or rng.random() < 0.6):
                    target = rng.choice(opts)
                elif acts:
                    target = rng.choice(acts)
                if target is None:
                    page.wait_for_timeout(200)
                    continue
                try:
                    target.click(timeout=2000)
                except Exception:  # the element went away under a new state: just retry
                    continue
                clicks += 1
                if clicks % args.shot_every == 0:
                    shots += 1
                    page.screenshot(path=str(OUT / f"{shots:02d}-play.png"))
                page.wait_for_timeout(120)
            page.screenshot(path=str(OUT / "99-end.png"))
            (OUT / "last-frames.json").write_text("\n".join(str(f) for f in frames[-3:]))
            browser.close()
        print(f"clicks={clicks} over={over} seconds={time.time() - t0:.0f} shots={shots}")
        for e in errors[:10]:
            print("ERROR", e)
        return 0 if over and not errors else 2
    finally:
        srv.terminate()
        srv.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
