"""Run a steps file against a page reached over CDP (a laptop Chromium, or the phone's WebView via adb forward).

    CDP=http://localhost:9444 OUT=data/cache/ui-drive uv run --with playwright python scripts/app/ui_drive/drive.py \
        scripts/app/ui_drive/play.py

Steps get `pg`, `shot(name)`, `tap(selector, i)`, `log`, `OUT`. play.py drives one game (MODE=duel|practice|puzzle,
VW/VH viewport, YOUDECK/BOTDECK, SEED) and runs audit.js at every step: text covered by another element of the same
layer, clipped by a non-scrolling ancestor without an ellipsis/clamp/fade, or off screen. NOTEBOOK 2026-11-03.
"""
import sys, json, time
from playwright.sync_api import sync_playwright
import os
OUT = os.environ.get("OUT", "data/cache/ui-drive")
os.makedirs(OUT, exist_ok=True)
logs = []
with sync_playwright() as pw:
    br = pw.chromium.connect_over_cdp(os.environ.get("CDP", "http://localhost:9444"))
    pg = br.contexts[0].pages[0]
    pg.on("console", lambda m: logs.append(f"console.{m.type}: {m.text}"))
    pg.on("pageerror", lambda e: logs.append(f"PAGEERROR: {e}"))
    def shot(name):
        pg.screenshot(path=f"{OUT}/{name}.png", scale="css")
    def tap(sel, i=0):
        ok = pg.evaluate("([s, i]) => { const e = document.querySelectorAll(s)[i]; if (!e) return false; e.scrollIntoView({block: 'center'}); e.click(); return true; }", [sel, i])
        if not ok: raise RuntimeError("no element " + sel)
        time.sleep(0.4)
    code = open(sys.argv[1]).read()
    try:
        exec(code, {"pg": pg, "shot": shot, "tap": tap, "time": time, "json": json, "log": logs, "OUT": OUT})
    finally:
        print("\n".join(l for l in logs if "error" in l.lower() or "PAGEERROR" in l or "warn" in l.lower())[:3000])
