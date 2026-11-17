"""Search-mode menu and Settings-jump checks (mock, touch): Describe tap selects, outside tap and back close the menu, "Download in Settings" opens Settings and flashes the panel. Run: uv run --no-project --with playwright==1.57.0 python -u scripts/app/popover_jump_check.py web <out_dir>"""
import sys, functools, http.server, threading
from playwright.sync_api import sync_playwright
web, out = sys.argv[1], sys.argv[2]
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=web); h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8796), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
errs = []; fails = []
def check(name, ok):
    print(("ok  " if ok else "FAIL") + " " + name); ok or fails.append(name)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2, has_touch=True, is_mobile=True)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    base = "http://127.0.0.1:8796/index.html?mock=1&i=0"
    pg.goto(base + "&ltab=decks&dbopen=my-snake-eye-3fa21c&dbview=add"); pg.wait_for_timeout(1500)
    pg.tap('[data-act="db:s-modemenu"]'); pg.wait_for_timeout(300)
    pg.tap('[data-act="db:s-mode"][data-smode="describe"]'); pg.wait_for_timeout(400)
    check("tap Describe selects it", "Describe" in pg.inner_text(".csch-mode") and not pg.query_selector(".csch-menu"))
    check("not-installed note shows", bool(pg.query_selector('[data-act="db:s-getdescribe"]')))
    pg.tap('[data-act="db:s-modemenu"]'); pg.wait_for_timeout(300)
    pg.tap(".dbhead .dbtitle"); pg.wait_for_timeout(300)
    check("outside tap closes mode menu", not pg.query_selector(".csch-menu"))
    pg.tap('[data-act="db:s-modemenu"]'); pg.wait_for_timeout(300)
    pg.go_back(); pg.wait_for_timeout(400)
    check("back closes mode menu, stays in editor", not pg.query_selector(".csch-menu") and bool(pg.query_selector(".dbedit")))
    pg.tap("[data-act=\"db:s-getdescribe\"]"); pg.wait_for_timeout(1200)
    check("Download in Settings opens Settings", bool(pg.query_selector("#settings")))
    bg = lambda: pg.evaluate("() => getComputedStyle(document.getElementById('set-describe')).backgroundColor")
    pg.wait_for_timeout(1500); b1 = bg()
    pg.tap('[data-act="dm:download"]'); pg.wait_for_timeout(800)   # mock download re-renders Settings every 400 ms
    b2 = bg(); pg.screenshot(path=out + "/jump-flash.png")
    pg.wait_for_timeout(1300); b3 = bg()
    pg.wait_for_timeout(1500); b4 = bg()
    print("bg", b1, "|", b2, "|", b3, "|", b4)
    a = lambda c: float(c.rstrip(")").split(",")[-1]) if c.count(",") == 3 else 1.0
    check("glow full during the hold and survives re-renders", a(b1) >= 0.139 and a(b2) >= 0.1)
    check("glow fading at ~4 s", b3 not in (b1,) and "rgba" in b3)
    check("glow gone at ~5.5 s", b4 in ("rgba(0, 0, 0, 0)", "rgba(184, 154, 94, 0)"))
    # model picker outside tap (Bot tab)
    pg.goto(base + "&ltab=play"); pg.wait_for_timeout(1500)
    mp = pg.query_selector('[data-pickopen="model"]')
    if mp:
        mp.tap(); pg.wait_for_timeout(300)
        opened = bool(pg.query_selector(".pickpanel"))
        pg.tap(".lobbyhead, h1, header"); pg.wait_for_timeout(300)
        check("model picker closes on outside tap (opened=%s)" % opened, not pg.query_selector(".pickpanel"))
    else:
        print("skip model picker (not shown in mock)")
    b.close()
print("errors", errs, "fails", fails)
