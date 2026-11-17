"""Back-step check in mock mode (Export sheet -> editor -> Decks tab -> Bot -> leaves; deck sheet closes). Run: uv run --no-project --with playwright==1.57.0 python -u scripts/app/back_steps_check.py web"""
import sys, functools, http.server, threading
from playwright.sync_api import sync_playwright
web = sys.argv[1]
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=web); h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8798), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
def state(pg):
    return pg.evaluate("""() => ({tab: (document.querySelector('[data-act=lobbytab][aria-selected=true], [data-act=lobbytab].sel, [data-act=lobbytab][aria-current]')||{}).textContent,
      editor: !!document.querySelector('.dbedit'), sheet: !!document.querySelector('.dbsheet, #decksheet, .decksheet'), len: history.length})""")
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 412, "height": 915})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://127.0.0.1:8798/index.html?mock=1&i=0"); pg.wait_for_timeout(1500)
    print("start", state(pg))
    pg.click('[data-act="lobbytab"][data-tab="decks"]'); pg.wait_for_timeout(500); print("decks tab", state(pg))
    pg.click('[data-act="db:open"]'); pg.wait_for_timeout(800); print("editor", state(pg))
    pg.click('[data-act="db:export"]'); pg.wait_for_timeout(400); print("export sheet", state(pg))
    for i in range(4):
        pg.go_back(); pg.wait_for_timeout(500); print("back", i + 1, state(pg), pg.url[-30:])
    pg.goto('http://127.0.0.1:8798/index.html?mock=1&i=0'); pg.wait_for_timeout(1500)
    pg.click('[data-pickopen="you"], [data-pickopen]:not([data-pickopen="model"])'); pg.wait_for_timeout(500); print("picker open", pg.evaluate("() => !!document.querySelector('[data-key^=deck], .decksheet, #decksheet')"))
    pg.go_back(); pg.wait_for_timeout(500); print("after back", pg.evaluate("() => !!document.querySelector('[data-key^=deck], .decksheet, #decksheet')"), state(pg), pg.url[-12:])
    b.close()
print("errors", errs)
