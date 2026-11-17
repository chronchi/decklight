"""Deck builder mock screenshots (mock mode, no server). Run: uv run --no-project --with playwright==1.57.0 python scripts/app/deckbuilder_shots.py web <out_dir>"""
import sys, subprocess, time
from playwright.sync_api import sync_playwright
web, out = sys.argv[1], sys.argv[2]
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8799", "--bind", "127.0.0.1", "-d", web],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
D = "my-snake-eye-3fa21c"
shots = [
    ("01-decks-tab", "ltab=decks"),
    ("02-editor-deck", f"ltab=decks&dbopen={D}"),
    ("13-deck-find", f"ltab=decks&dbopen={D}&dbq=fire+king"),
    ("14-deck-find-text", f"ltab=decks&dbopen={D}&dbq=negate&dbchip=monster"),
    ("03-editor-pick", f"ltab=decks&dbopen={D}&dbpick=main:9674034"),
    ("04-add-search", f"ltab=decks&dbopen={D}&dbview=add&dbq=snake"),
    ("05-add-text-hit", f"ltab=decks&dbopen={D}&dbview=add&dbq=negate"),
    ("06-add-chips", f"ltab=decks&dbopen={D}&dbview=add&dbchip=trap"),
    ("07-describe", f"ltab=decks&dbopen={D}&dbview=add&dbmode=describe"),
    ("08-export", f"ltab=decks&dbopen={D}&dbsheet=export"),
    ("09-import", "ltab=decks&dbsheet=import"),
    ("10-menu", f"ltab=decks&dbopen={D}&dbsheet=menu"),
    ("11-card", f"ltab=decks&dbopen={D}&dbsheet=card&dbcard=9674034"),
    ("12-ryzeal-illegal", "ltab=decks&dbopen=ryzeal-test-8b01d2"),
]
errs = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for vw, vh, tag in [(412, 915, ""), (360, 740, "-360")]:
        ctx = b.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=2, is_mobile=True, has_touch=True)
        for name, q in shots:
            if tag and name not in ("01-decks-tab", "02-editor-deck", "04-add-search"):
                continue
            pg = ctx.new_page()
            pg.on("pageerror", lambda e, n=name: errs.append(f"{n}: {e}"))
            pg.on("console", lambda m, n=name: m.type == "error" and errs.append(f"{n}: console {m.text}"))
            pg.goto(f"http://127.0.0.1:8799/index.html?mock=1&i=0&{q}")
            pg.wait_for_timeout(900)
            pg.screenshot(path=f"{out}/{name}{tag}.png")
            pg.close()
    b.close()
srv.terminate()
print("errors:", errs or "none")
