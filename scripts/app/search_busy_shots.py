"""Search busy state + Stop, and the subsection headings (mock). Run: uv run --no-project --with playwright==1.57.0 python -u scripts/app/search_busy_shots.py web <out_dir>
Exits 1 when a check fails: spinner + Stop while searching, Search back after Stop, results arrive when not stopped."""
import sys, functools, http.server, threading
from playwright.sync_api import sync_playwright
web, out = sys.argv[1], sys.argv[2]
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=web); h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8798), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
errs, bad = [], []
def check(ok, what): print(("ok   " if ok else "FAIL ") + what, flush=True); ok or bad.append(what)
U = "http://127.0.0.1:8798/index.html?mock=1&i=0&dmstate=installed&ltab=decks&dbopen=my-snake-eye-3fa21c&dbview=add&dbmode=describe"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    for run in ("stop", "finish"):
        pg.goto(U); pg.wait_for_timeout(1200)
        pg.fill(".dbedit input[type=search]", "a monster that destroys spells"); pg.click(".csch-go"); pg.wait_for_timeout(800)
        check(pg.locator(".csch-spin").count() == 1 and pg.locator(".csch-go.stop").count() == 1, run + ": spinner and Stop shown")
        if run == "stop":
            pg.screenshot(path=f"{out}/sb1-searching.png")
            pg.click(".csch-go.stop"); pg.wait_for_timeout(4500)
            check(pg.locator(".csch-spin").count() == 0 and pg.locator(".csch-r").count() == 0 and
                  pg.locator(".csch-go:not(.stop)").count() == 1, "stop: back to Search, late reply ignored")
            pg.screenshot(path=f"{out}/sb2-stopped.png")
        else:
            pg.wait_for_timeout(4000)
            check(pg.locator(".csch-r").count() > 0 and pg.locator(".csch-spin").count() == 0, "finish: results shown")
            pg.screenshot(path=f"{out}/sb3-results.png")
    for name, q in [("hd1-settings", "settings=describe&dmstate=installed"), ("hd2-lobby", "ltab=bot")]:
        pg.goto("http://127.0.0.1:8798/index.html?mock=1&i=0&" + q); pg.wait_for_timeout(1500)
        pg.screenshot(path=f"{out}/{name}.png")
    b.close()
print("errors", errs)
sys.exit(1 if bad or errs else 0)
