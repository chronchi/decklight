"""Screenshot the phone-first web client against mock states.

Serves web/ with a tiny static server, opens ?mock=1&i=N at 412x915
(device scale 2, isMobile) saving one PNG per mock to
data/cache/web-screens/, plus a 360x780 small-phone variant of 02
and one desktop shot at 1280x800.

Fails (exit 1) on any page error or console error. Prints measured
card width and board-to-hand gap at both phone sizes.
"""

import functools
import http.server
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
OUT = ROOT / "data" / "cache" / "web-screens"

MOCKS = [
    ("01-lobby", 0),
    ("02-your-main-phase", 1),
    ("03-chain-prompt", 2),
    ("04-select-cards", 3),
    ("05-bot-thinking", 4),
    ("06-battle", 5),
    ("07-position", 6),
    ("08-game-over", 7),
    ("09-gy-sheet", 8),
    ("10-chain-3", 9),
    ("11-focus", "1&focus=ym2"),
    ("12-menu", "1&menu=yh2"),
    ("13-lobby-picker", "0&picker=you"),
    ("14-select-picked", "3&sel=1,3"),
]


def launch_kwargs():
    """Point at the pre-installed Chromium (brief: already in ~/.cache/ms-playwright)."""
    env = os.environ.get("CHROME_EXE")
    if env:
        return {"executable_path": env}
    ms = Path.home() / ".cache" / "ms-playwright"
    for cand in sorted(ms.glob("chromium-*/chrome-linux*/chrome"), reverse=True):
        return {"executable_path": str(cand)}
    return {}


def serve():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(WEB))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


MEASURE_JS = """() => {
  const r = (el) => el ? el.getBoundingClientRect() : null;
  const round = (x) => Math.round(x * 10) / 10;
  const main = r(document.querySelector('.field.you [data-row="mzone"] .slot:not(.side)'));
  const side = r(document.querySelector('.field.you [data-row="szone"] .slot.side'));
  const hc = r(document.querySelector('#hand .hcard'));
  // Vertical blocks top to bottom; dead gap = space between consecutive blocks.
  const blocks = ['.pbar.bot', '#board', '.pbar.you', '#info', '#hand'].map((q) => [q, r(document.querySelector(q))]);
  const gaps = [];
  for (let i = 0; i + 1 < blocks.length; i++) gaps.push(round(blocks[i + 1][1].top - blocks[i][1].bottom));
  // Empty space inside the hand block above the cards (minus the 10px lift room)
  // and below the last feed row in the info panel.
  const hand = r(document.querySelector('#hand'));
  const handSlack = hc ? round(hc.top - hand.top - 14) : -1;
  const evs = document.querySelectorAll('#info .ev');
  const info = r(document.querySelector('#info'));
  const infoSlack = evs.length ? round(info.bottom - evs[evs.length - 1].getBoundingClientRect().bottom) : -1;
  const docW = document.documentElement.scrollWidth;
  return {boardW: main && round(main.width), sideW: side && round(side.width), handW: hc && round(hc.width),
    handH: hc && round(hc.height), infoH: info && round(info.height), gaps, handSlack, infoSlack,
    overflowX: docW > window.innerWidth, bottom: round(hand.bottom), vh: window.innerHeight};
}"""


def main() -> int:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    httpd, port = serve()
    base = f"http://127.0.0.1:{port}/index.html"
    numbers: dict[str, object] = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs())
            # Phone shots, one per mock.
            ctx = browser.new_context(viewport={"width": 412, "height": 915},
                                      device_scale_factor=2, is_mobile=True)
            page = ctx.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console-{m.type}: {m.text}")
                      if m.type in ("error",) else None)
            for name, i in MOCKS:
                page.goto(f"{base}?mock=1&i={i}", wait_until="networkidle")
                page.wait_for_timeout(400)
                page.screenshot(path=str(OUT / f"{name}.png"))
                print(f"shot {name}.png", flush=True)
                if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
                    errors.append(f"horizontal overflow at 412px in {name}")
                if name == "02-your-main-phase":
                    numbers["m412"] = page.evaluate(MEASURE_JS)
            ctx.close()
            # Small-phone variant of 02.
            ctx3 = browser.new_context(viewport={"width": 360, "height": 780},
                                       device_scale_factor=2, is_mobile=True)
            s = ctx3.new_page()
            s.on("pageerror", lambda e: errors.append(f"pageerror-small: {e}"))
            s.on("console", lambda m: errors.append(f"console-small-{m.type}: {m.text}")
                   if m.type in ("error",) else None)
            for nm, i in [("02-small", 1), ("03-small", 2), ("01-small", 0), ("07-small", 6)]:
                s.goto(f"{base}?mock=1&i={i}", wait_until="networkidle")
                s.wait_for_timeout(400)
                s.screenshot(path=str(OUT / f"{nm}.png"))
                print(f"shot {nm}.png", flush=True)
                if nm == "02-small":
                    numbers["m360"] = s.evaluate(MEASURE_JS)
                if s.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
                    errors.append(f"horizontal overflow at 360px in {nm}")
            ctx3.close()
            # Desktop shot of the main-phase board (log drawer open).
            ctx2 = browser.new_context(viewport={"width": 1280, "height": 800})
            d = ctx2.new_page()
            d.on("pageerror", lambda e: errors.append(f"pageerror-desktop: {e}"))
            d.on("console", lambda m: errors.append(f"console-desktop-{m.type}: {m.text}")
                       if m.type in ("error",) else None)
            d.goto(f"{base}?mock=1&i=1", wait_until="networkidle")
            d.wait_for_timeout(400)
            d.screenshot(path=str(OUT / "09-desktop.png"))
            print("shot 09-desktop.png", flush=True)
            ctx2.close()
            browser.close()
    finally:
        httpd.shutdown()
    print(f"measure412: {numbers.get('m412')}", flush=True)
    print(f"measure360: {numbers.get('m360')}", flush=True)
    if errors:
        print("ERRORS:", flush=True)
        for e in errors:
            print(f"  {e}", flush=True)
        return 1
    print(f"ok: {len(MOCKS) + 5} PNGs in {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
