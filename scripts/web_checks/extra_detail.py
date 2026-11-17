# Browser check (user 2026-11-11: Extra Deck cards showed no details): `uv run --with playwright python scripts/web_checks/extra_detail.py out.png`.
import sys; sys.path.insert(0, "scripts")
from web_screens import launch_kwargs
import functools, http.server, threading
from playwright.sync_api import sync_playwright
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory="web")
h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(**launch_kwargs()); pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2, is_mobile=True, has_touch=True)
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/index.html?mock=1&i=1&pile=extra:you"); pg.wait_for_timeout(800)
    cells = pg.locator("#sheet.pilesheet .cell"); print("cells", cells.count(), "backs", pg.locator("#sheet.pilesheet .card.back").count())
    cells.first.click(); pg.wait_for_timeout(400)
    print("detail open:", pg.locator("#detail").count(), pg.locator("#detail").first.inner_text()[:80].replace("\n"," | ") if pg.locator("#detail").count() else "")
    pg.screenshot(path=sys.argv[1]); print("errors", errs)
