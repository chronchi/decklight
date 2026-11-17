# Browser check (user 2026-11-11: the sheets could not be pushed down): swipe-down on a pile sheet / the duel log closes it.
# `uv run --with playwright python scripts/web_checks/swipe_close.py`.
import sys; sys.path.insert(0, "scripts")
from web_screens import launch_kwargs
import functools, http.server, threading
from playwright.sync_api import sync_playwright
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory="web"); h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(**launch_kwargs()); pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2, is_mobile=True, has_touch=True)
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    base = f"http://127.0.0.1:{srv.server_address[1]}/index.html?mock=1&i=1"
    for q, sel in (("&pile=extra:you", "#sheet.pilesheet"), ("&log=1", "#logdrawer")):
        pg.goto(base + q); pg.wait_for_timeout(700)
        el = pg.locator(sel); print(sel, "open", el.count())
        if not el.count(): continue
        bb = el.bounding_box()
        x, y = bb["x"] + bb["width"] / 2, bb["y"] + 10
        pg.mouse.move(x, y); pg.mouse.down()
        for i in range(1, 11): pg.mouse.move(x, y + 15 * i)
        pg.mouse.up(); pg.wait_for_timeout(400)
        print(sel, "after swipe", pg.locator(sel).count())
    print("errors", errs)
