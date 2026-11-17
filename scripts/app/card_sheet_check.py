"""Card sheet from a search row: text loads, Main/Extra + Side steppers work, 3-copy cap (mock).
Run: uv run --no-project --with playwright==1.57.0 python -u scripts/app/card_sheet_check.py web <out_dir>"""
import sys, functools, http.server, threading
from playwright.sync_api import sync_playwright
web, out = sys.argv[1], sys.argv[2]
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8799), functools.partial(Q, directory=web))
threading.Thread(target=srv.serve_forever, daemon=True).start()
errs, bad = [], []
def check(ok, what): print(("ok   " if ok else "FAIL ") + what, flush=True); ok or bad.append(what)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://127.0.0.1:8799/index.html?mock=1&i=0&ltab=decks&dbopen=my-snake-eye-3fa21c&dbview=add"); pg.wait_for_timeout(1500)
    pg.fill(".dbedit input[type=search]", "e"); pg.wait_for_timeout(800)
    pg.locator(".csch-r:not(:has(.csch-add2 b)) .csch-open").first.click(); pg.wait_for_timeout(600)
    check("Loading" not in pg.inner_text(".dbcdtext"), "card text loaded from a search row")
    rows = [r.inner_text().split("\n")[0] for r in pg.locator(".dbsecrow").all()]
    check(len(rows) == 2 and rows[0] in ("Main Deck", "Extra Deck") and rows[1] == "Side Deck", "two stepper rows: " + str(rows))
    plus = pg.locator(".dbsecrow").nth(0).locator("[data-act='db:cplus']")
    for _ in range(2): plus.click(); pg.wait_for_timeout(150)
    pg.locator(".dbsecrow").nth(1).locator("[data-act='db:cplus']").click(); pg.wait_for_timeout(150)
    ns = [r.locator(".n").inner_text() for r in pg.locator(".dbsecrow").all()]
    check(ns == ["2", "1"], "counts 2 + 1: " + str(ns))
    check(plus.is_disabled(), "+ disabled at 3 copies")
    pg.screenshot(path=f"{out}/cs1-sheet.png"); b.close()
print("errors", errs); sys.exit(1 if bad or errs else 0)
