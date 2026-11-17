"""The app at every width class (docs/plans/responsive.md; user 2026-11-16: "Rail"): phone, foldable, tablet portrait and
landscape, 13" and 16" laptops. On each it opens Games (the list, then a game: two panes from 924 px), Ask and a live
duel, saves a screenshot, and flags:
  - the bottom bar on a screen of 600 px or more, or the rail on a phone;
  - a page wider than the screen (sideways scroll);
  - Games showing one pane where two fit, or two where they do not;
  - a duel whose cards did not grow on a large screen (--card-w at the phone's 76 px cap or less);
  - from 1100 px, a duel without its three columns: the card reader left of the table (showing a card, never empty) and
    the log right of it (docs/ui/mocks/table/).
Games come from `writes_demo_games` (see games_check.py). Run from ygo-zero/ with ORT_DYLIB_PATH set:
    uv run --no-project --with playwright==1.57.0 python -u scripts/app/responsive_check.py <dir>/records <out_dir>"""
import shutil, subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path.cwd(); records = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
if not (records / "index.sqlite").is_file():
    sys.exit(f"no saved games in {records}: run writes_demo_games first (see games_check.py)")
shutil.rmtree(out / "demo", ignore_errors=True); (out / "demo").mkdir()
shutil.copy(records.parent / "install_id", out / "demo" / "install_id")
records = Path(shutil.copytree(records, out / "demo" / "records", ignore=shutil.ignore_patterns("trash", "index.sqlite*")))
(records / "index.sqlite").touch()
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8836", "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(records)], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
DEVICES = {"phone": (412, 915), "fold": (773, 900), "tab-p": (820, 1180), "tab-l": (1180, 820), "13in": (1280, 720), "16in": (1728, 1035)}
problems, errs = [], []


def check(pg, dev, w, where):
    sw = pg.evaluate("document.documentElement.scrollWidth")
    if sw > w + 1:
        problems.append(f"{dev} {where}: the page is {sw} px wide on a {w} px screen")
    bar = pg.locator("#lobby > .bbar")
    if bar.count():
        r = bar.bounding_box()
        rail = r["height"] > r["width"]
        if w >= 600 and not rail:
            problems.append(f"{dev} {where}: a bottom bar at {w} px (the rail starts at 600)")
        if w < 600 and rail:
            problems.append(f"{dev} {where}: a rail on a phone")


try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for dev, (w, h) in DEVICES.items():
            pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto("http://127.0.0.1:8836/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
            pg.click('.lsub button[data-tab="games"]'); pg.wait_for_selector(".gm-row", timeout=30000)
            check(pg, dev, w, "games list"); pg.screenshot(path=str(out / f"{dev}-1-list.png"))
            pg.locator(".gm-row:not(:has(.gm-s))").first.click(); pg.wait_for_selector('[data-act="games:watch"]', timeout=30000)
            pg.wait_for_timeout(500)
            two = pg.locator(".gm.two").count() > 0
            if two != (w >= 924):
                problems.append(f"{dev}: Games shows {'two panes' if two else 'one pane'} at {w} px (two from 924)")
            check(pg, dev, w, "game"); pg.screenshot(path=str(out / f"{dev}-2-game.png"))
            pg.click('[data-act="games:askgame"]'); pg.wait_for_selector("#akin", timeout=30000)
            check(pg, dev, w, "ask"); pg.screenshot(path=str(out / f"{dev}-3-ask.png"))
            pg.close()
        # The live duel last: the server keeps it, so a new page lands back in it at each size.
        pg = b.new_page(viewport={"width": 412, "height": 915})
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto("http://127.0.0.1:8836/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
        pg.click(".bbar button:first-child"); pg.click('.lsub button[data-tab="play"]')
        pg.wait_for_selector('#startbtn:not([disabled])', timeout=60000); pg.click("#startbtn")
        for _ in range(12):
            if pg.locator('[data-act="gamemenu"]').count(): break
            for sel in ('[data-act="rpspick"]', '[data-act="rpsagain"]', '[data-act="rpsfirst"]', '[data-act="rpsbotfirst"]'):
                if pg.locator(sel).count(): pg.locator(sel).first.click(); break
            pg.wait_for_timeout(1500)
        pg.wait_for_selector('[data-act="gamemenu"]', timeout=60000)
        for dev, (w, h) in DEVICES.items():
            pg.set_viewport_size({"width": w, "height": h}); pg.wait_for_timeout(700)
            cw = float(pg.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--card-w')").strip().rstrip("px") or 0)
            print(dev, "card", cw, flush=True)
            if dev in ("tab-p", "16in") and cw <= 76:
                problems.append(f"{dev}: the duel's cards stayed at {cw} px")
            if w >= 1100:
                rd, ap, lg = (pg.locator(s).bounding_box() for s in ("#readside", "#app", "#logside"))
                if not rd or rd["width"] < 280 or rd["x"] + rd["width"] > ap["x"] + 1 or lg["x"] < ap["x"] + ap["width"] - 1:
                    problems.append(f"{dev}: the duel is not reader | table | log ({rd}, {ap}, {lg})")
                elif not pg.locator("#readside h2").count():
                    problems.append(f"{dev}: the card reader is empty")
            elif pg.locator("#readside").is_visible():
                problems.append(f"{dev}: the card reader column shows at {w} px (from 1100)")
            check(pg, dev, w, "duel"); pg.screenshot(path=str(out / f"{dev}-4-duel.png"))
        b.close()
except Exception as e:  # noqa: BLE001
    problems.append(f"driver: {str(e).splitlines()[0]}")
finally:
    srv.kill()
for x in problems:
    print(x)
print("page errors:", errs)
print(len(problems), "problems")
sys.exit(1 if problems or errs else 0)
