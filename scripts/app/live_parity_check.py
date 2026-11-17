"""Today's client features against the real server on the laptop (not the mock): deck picker tabs, starters, examples,
filters (attribute/race/ATK and Spell type), card sheet text, + Side. Exits 1 on any failed check or page error.
Run from ygo-zero/: uv run --no-project --with playwright==1.57.0 python -u scripts/app/live_parity_check.py $PWD <out_dir>
Seeds decks/user/ (git-ignored) with the two starter decks on first run; the smoke deck it creates is deleted at the end."""
import re, sys, time, subprocess, pathlib
from playwright.sync_api import sync_playwright

root, out = sys.argv[1], pathlib.Path(sys.argv[2])
srv = subprocess.Popen([root + "/rust/target/release/decklight-serve", "--port", "8812", "--token", "parity"], cwd=root,
                       stdout=open(out / "parity-server.log", "w"), stderr=subprocess.STDOUT)
errs, bad = [], []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        bad.append(what)


def lines(pg):
    return [t for t in pg.locator(".csch-r .ln").all_inner_texts()]


try:
    time.sleep(5)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: m.type == "error" and "404" not in m.text and errs.append(m.text))
        pg.goto("http://127.0.0.1:8812/?t=parity&ltab=bot"); pg.wait_for_timeout(3000)

        # Deck picker: Yours (starters) / Prebuilt (22)
        pg.locator("[data-pickopen='you']").first.click(); pg.wait_for_timeout(800)
        tabs = pg.locator(".dksrc button").all_inner_texts()
        print("     tabs:", tabs)
        names = pg.locator(".dkname").all_inner_texts()
        subs = pg.locator(".dkrow").first.inner_text()
        check(pg.locator(".dkinfo").count() >= 2 and ("Your deck" in subs or "Not legal" in subs),
              "Yours rows use the Prebuilt layout (summary line + deck list button): " + subs.replace("\n", " | "))
        pg.click("[data-dsrc='prebuilt']"); pg.wait_for_timeout(500)
        # Prebuilt = the 22 curated decks + one per fine-tuned model found in runs/ (the phone has none, so 22 there).
        tuned = sum("fine-tuned" in t.lower() for t in pg.locator(".dksub").all_inner_texts())
        n_pre = int(re.search(r"Prebuilt\s+(\d+)", " ".join(tabs)).group(1))
        check(any(t.startswith("Yours") for t in tabs) and n_pre == 22 + tuned,
              f"picker tabs Yours / Prebuilt {n_pre} = 22 curated + {tuned} fine-tuned")
        check("Blue-Eyes" in names and "Branded" in names, "Yours lists the starters: " + str(names[:4]))
        pg.screenshot(path=str(out / "p1-picker.png"))
        pg.click("#decksheet [data-act='closedecks']"); pg.wait_for_timeout(400)

        # Deck builder: a fresh deck, Add cards, examples
        pg.goto("http://127.0.0.1:8812/?t=parity&ltab=decks"); pg.wait_for_timeout(2500)
        pg.click("[data-act='db:new']"); pg.wait_for_timeout(1500)
        if pg.locator(".exbtn").count() == 0:
            pg.click('[data-act="db:view"][data-view="add"]'); pg.wait_for_timeout(1200)
        check(pg.locator(".exbtn").count() == 3, "three examples in the empty search")
        pg.locator(".exbtn", has_text="Ash Blossom").click(); pg.wait_for_timeout(2500)
        top = pg.locator(".csch-r .nm").all_inner_texts()[:3]
        check(bool(top) and "Ash Blossom" in top[0], "example 'Ash Blossom' finds it first: " + str(top))

        # Filters: LIGHT Dragon ATK >= 3000
        pg.click("[data-act='db:s-clear']"); pg.wait_for_timeout(800)
        pg.click(".csch-fbtn"); pg.wait_for_timeout(400)
        pg.click("[data-fv='LIGHT']"); pg.click("[data-fv='Dragon']")
        pg.fill("[data-dbinput='f:atk:0']", "3000"); pg.press("[data-dbinput='f:atk:0']", "Tab"); pg.wait_for_timeout(200)
        pg.click("[data-act='db:s-fapply']"); pg.wait_for_timeout(2500)
        ls = lines(pg)
        status = pg.locator(".csch-status").inner_text() if pg.locator(".csch-status").count() else ""
        print("     status:", status, "| first:", ls[:2])
        atk_ok = all(int(m.group(1)) >= 3000 for l in ls for m in [re.search(r"(\d+|\?)/", l) or re.search(r"· (\d+)$", l)]
                     if m and m.group(1).isdigit())
        check(len(ls) > 0 and all("LIGHT" in l and "Dragon" in l for l in ls) and atk_ok,
              f"filters LIGHT+Dragon+ATK>=3000: {len(ls)} rows all match")
        check(pg.locator(".csch-fbtn b").inner_text() == "3", "Filters chip shows 3")
        pg.screenshot(path=str(out / "p2-filters.png"))

        # Spell type: Quick-Play spells only
        pg.click(".csch-fbtn"); pg.click("[data-act='db:s-fclear']"); pg.click("[data-fv='quickplay']")
        pg.click("[data-act='db:s-fapply']"); pg.wait_for_timeout(2500)
        ls = lines(pg)
        print("     quick-play first:", ls[:2])
        check(len(ls) > 0 and all("Quick" in l for l in ls), f"st_type quickplay: {len(ls)} rows all Quick-Play")

        # Card sheet text + + Side
        pg.click(".csch-fbtn"); pg.click("[data-act='db:s-fclear']"); pg.click("[data-act='db:s-fapply']")
        pg.fill('[data-dbinput="q"]', "ash blossom"); pg.wait_for_timeout(2500)
        row = pg.locator(".csch-r").first
        row.locator("[data-sec='side']").click(); pg.wait_for_timeout(800)
        check(row.locator("[data-sec='side'] b").inner_text() == "1", "+ Side adds one to Side")
        row.locator(".csch-open").click(); pg.wait_for_timeout(1500)
        txt = pg.locator(".dbcdtext").inner_text()
        check("Loading" not in txt and len(txt) > 40, "card sheet shows the card text")
        rows = [r.split("\n")[0] for r in pg.locator(".dbsecrow").all_inner_texts()]
        check(rows == ["Main Deck", "Side Deck"], "sheet steppers Main/Side: " + str(rows))
        pg.screenshot(path=str(out / "p3-sheet.png"))
        pg.click(".dbsheet [data-act='db:closesheet']"); pg.wait_for_timeout(300)

        # Delete the smoke deck
        pg.click("[data-act='db:menu']"); pg.wait_for_timeout(300); pg.click("[data-act='db:askdelete']"); pg.wait_for_timeout(300)
        pg.click("[data-act='db:delete']"); pg.wait_for_timeout(1200)
        b.close()
finally:
    srv.terminate()
print("errors", errs)
sys.exit(1 if bad or errs else 0)
