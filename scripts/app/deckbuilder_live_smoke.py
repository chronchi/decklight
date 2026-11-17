"""Deck builder live smoke against decklight-serve (import, deck filter, typo search, add, legality).
Run from ygo-zero/: uv run --no-project --with playwright==1.57.0 python -u scripts/app/deckbuilder_live_smoke.py $PWD <out_dir> decks/custom/wiccanthrope.ydk
Writes decks/user/ (git-ignored); delete it afterwards."""
import sys, time, subprocess, pathlib
from playwright.sync_api import sync_playwright
root, out, ydk = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]).read_text()
srv = subprocess.Popen([root + "/rust/target/release/decklight-serve", "--port", "8811", "--token", "smoke"], cwd=root,
                       stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT)
errs = []
try:
    time.sleep(4)
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
        pg.goto("http://127.0.0.1:8811/?t=smoke&ltab=decks"); pg.wait_for_timeout(2500)
        pg.screenshot(path=str(out / "l1-list.png"))
        pg.click('[data-act="db:import"]'); pg.wait_for_timeout(400)
        pg.fill('[data-dbinput="imptext"]', ydk); pg.fill('[data-dbinput="impname"]', "Smoke Wiccanthrope")
        pg.click('[data-act="db:doimport"]'); pg.wait_for_timeout(2500)
        pg.screenshot(path=str(out / "l2-imported.png"))
        q = pg.locator('[data-dbinput="q"]')
        q.fill("special summon"); pg.wait_for_timeout(800); pg.screenshot(path=str(out / "l3-deckfind.png"))
        pg.click('[data-act="db:view"][data-view="add"]'); pg.wait_for_timeout(1500)
        pg.screenshot(path=str(out / "l4-add-carried.png"))
        q = pg.locator('[data-dbinput="q"]'); q.fill("ash blosom"); pg.wait_for_timeout(4000)
        pg.screenshot(path=str(out / "l5-typo.png"))
        pg.locator('[data-act="db:s-add"]').first.click(); pg.wait_for_timeout(1500)
        pg.screenshot(path=str(out / "l6-added.png"))
        pg.click('[data-act="db:close"]'); pg.wait_for_timeout(1500)
        pg.screenshot(path=str(out / "l7-list-after.png"))
        b.close()
finally:
    srv.terminate(); srv.wait()
print("errors:", errs)
