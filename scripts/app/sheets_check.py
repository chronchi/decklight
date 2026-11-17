"""Every bottom sheet in the app closes on a swipe down from its grab bar, Escape and a tap outside (user 2026-11-15:
"make sure this behavior is consistent across the whole app"): Ask's model sheet, the deck builder's sheets, the game
menu. Run from ygo-zero/ with ORT_DYLIB_PATH set (the duel needs the bot; ort rc.15 wants ONNX Runtime >= 1.31, e.g.
the pip wheel's: `uv run --no-project --python 3.12 --with "onnxruntime>=1.31" python -c "import onnxruntime; print(
onnxruntime.__file__)"` → capi/libonnxruntime.so.1.32.0):
    uv run --no-project --with playwright==1.57.0 python -u scripts/app/sheets_check.py <out_dir>"""
import subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path.cwd(); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8816", "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(out / "records")], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
res, errs = [], []
def swipe(pg, sel):
    # The sheet slides up when it opens: the bar measured mid-slide is lower than where the press lands, so the press
    # missed the header band and the deck-import sheet "STAYED OPEN" (2026-11-15). Same wait as the touch swipe.
    pg.wait_for_timeout(400)
    g = pg.locator(sel + " .grab").first.bounding_box(); x, y = g["x"] + g["width"] / 2, g["y"] + 2
    pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(x, y + 60, steps=5); pg.mouse.move(x, y + 200, steps=5); pg.mouse.up()
def touch_swipe(pg, sel):
    # A finger, not a mouse: CDP touch events go through the browser's pan handling, which cancels the pointer
    # (pointercancel) when the header can scroll. The mouse swipe above never hits that (2026-11-15: it passed while
    # the swipe did nothing on the phone). Starts a few px beside the 4 px bar, where a finger lands.
    pg.wait_for_timeout(400)   # the sheet slides up when it opens: measure the bar where it stops
    g = pg.locator(sel + " .grab").first.bounding_box(); x, y = g["x"] + g["width"] / 2 + 20, g["y"] + 8
    cdp = pg.context.new_cdp_session(pg)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    for i in range(1, 13):
        cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x, "y": y + 18 * i}]})
        pg.wait_for_timeout(16)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
def audit(pg, name, opener, sel):
    for how in ("swipe", "touch", "escape", "outside"):
        opener()
        try:
            pg.wait_for_selector(sel, timeout=15000)
        except Exception:
            res.append((name, how, "did not open")); continue
        if how == "swipe": swipe(pg, sel)
        elif how == "touch": touch_swipe(pg, sel)
        elif how == "escape": pg.keyboard.press("Escape")
        else: pg.mouse.click(206, 40)
        pg.wait_for_timeout(500)
        res.append((name, how, "ok" if not pg.locator(sel).count() else "STAYED OPEN"))
        pg.screenshot(path=str(out / f"{name}-{how}.png"))
try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 412, "height": 915}, has_touch=True)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto("http://127.0.0.1:8816/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
        pg.click('[data-tab="ask"]')
        audit(pg, "ask-model", lambda: pg.click('.ak-model'), ".ak-bsheet")
        def open_provider():
            if not pg.locator(".ak-prow").count(): pg.click('.ak-model'); pg.wait_for_selector(".ak-prow")
            pg.click(".ak-prow")
        audit(pg, "ask-provider", open_provider, '.ak-bsheet[aria-label="Provider"]')
        pg.keyboard.press("Escape"); pg.wait_for_timeout(300); pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
        pg.click('[data-tab="decks"]')
        audit(pg, "deck-import", lambda: pg.click('[data-act="db:import"]'), ".dbsheet")
        pg.click('[data-tab="play"]')
        pg.wait_for_selector('#startbtn:not([disabled])', timeout=60000); pg.click("#startbtn")
        for _ in range(10):
            if pg.locator('[data-act="gamemenu"]').count(): break
            for sel in ('[data-act="rpspick"]', '[data-act="rpsagain"]', '[data-act="rpsfirst"]', '[data-act="rpsbotfirst"]'):
                if pg.locator(sel).count(): pg.locator(sel).first.click(); break
            pg.wait_for_timeout(1500)
        try:
            pg.wait_for_selector('[data-act="gamemenu"]', timeout=60000)
        except Exception:
            pg.screenshot(path=str(out / "start.png")); raise
        audit(pg, "game-menu", lambda: pg.click('[data-act="gamemenu"]'), "#gamemenu")
        b.close()
except Exception as e:
    res.append(("driver", str(e).splitlines()[0], "FAIL"))
finally:
    srv.kill()
for r in res: print(*r)
print("page errors:", errs)
sys.exit(0 if all(r[2] == "ok" for r in res) and not errs else 1)
