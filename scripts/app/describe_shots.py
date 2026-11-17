"""Describe-a-card Settings panel screenshots (mock). Run: uv run --no-project --with playwright==1.57.0 python -u scripts/app/describe_shots.py web <out_dir>"""
import sys, functools, http.server, threading
from playwright.sync_api import sync_playwright
web, out = sys.argv[1], sys.argv[2]
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=web); h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8797), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
errs = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, q, wait in [("dm1-idle", "settings=describe", 1200), ("dm2-downloading", "settings=describe&dmstate=downloading", 3000),
                          ("dm3-installed", "settings=describe&dmstate=downloading", 12000)]:
        pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(f"http://127.0.0.1:8797/index.html?mock=1&i=0&{q}"); pg.wait_for_timeout(wait)
        pg.screenshot(path=f"{out}/{name}.png"); pg.close()
    b.close()
print("errors", errs)
