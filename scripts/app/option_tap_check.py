"""A tap on an option's effect line chooses the option when the line is shown whole (user 2026-11-13: Mulcharmy
Fuwalos "didn't work" against Sky Striker Mobilize - Linkage!: the tap landed on its effect text, which only toggled
expansion). Exits 1 when the tap does not choose or a page error occurs.
Run from ygo-zero/: uv run --no-project --with playwright==1.57.0 python -u scripts/app/option_tap_check.py <out>
It serves a copy of web/ in <out> whose mock chain prompt is scripts/app/fixtures/fuwalos-chain.json (the frame of the
phone's record 01a1d71b at response 18, built with decklight.web.view build_view/build_prompt)."""
import shutil, sys, subprocess, time
from playwright.sync_api import sync_playwright

out = sys.argv[1]
web = out + "/web"
shutil.rmtree(web, ignore_errors=True)
shutil.copytree("web", web)
shutil.copy("scripts/app/fixtures/fuwalos-chain.json", web + "/mock/03-chain-prompt.json")
srv = subprocess.Popen(["python3", "-m", "http.server", "8814", "-d", web], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
errs, bad = [], []
try:
    time.sleep(1)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2, has_touch=True)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto("http://127.0.0.1:8814/?mock=1&i=2"); pg.wait_for_timeout(2500)
        ef = pg.locator("#sheet .opt", has_text="Fuwalos").locator(".ef")
        ef.tap(); pg.wait_for_timeout(100)
        chosen = "locked" in (pg.locator("#hand").get_attribute("class") or "")
        print(("ok   " if chosen else "FAIL ") + "tap on Fuwalos's effect text chooses it", flush=True)
        if not chosen:
            bad.append("tap")
        pg.screenshot(path=out + "/after-tap.png")
        b.close()
finally:
    srv.terminate()
print("errors", errs)
sys.exit(1 if bad or errs else 0)
