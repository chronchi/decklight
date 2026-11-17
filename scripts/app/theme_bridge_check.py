"""Run from ygo-zero/: uv run --no-project --with playwright==1.57.0 python -u scripts/app/theme_bridge_check.py <out_dir>
The theme bridge as the Android app gives it: the saved mode wins over an empty localStorage, System follows the
bridge (not prefers-color-scheme), and a phone switch (__decklightUiMode) recolours."""
import subprocess, time, shutil, sys
from pathlib import Path
from playwright.sync_api import sync_playwright
out = Path(sys.argv[1]); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
srv = subprocess.Popen(["rust/target/release/decklight-serve", "--port", "8812", "--token", "smoke", "--ask-dir", str(out / "ask"), "--records", str(out / "rec")], stdout=open(out / "log", "wb"), stderr=subprocess.STDOUT)
BR = """window.__applied = []; window.__sys = '%s'; window.DecklightTheme = { mode: () => '%s', system: () => window.__sys,
  apply: (m, t) => window.__applied.push(m + ':' + t) };"""
bad = []
try:
    time.sleep(3)
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        def page(sysv, mode, scheme="light"):
            pg = b.new_page(color_scheme=scheme); pg.add_init_script(BR % (sysv, mode))
            pg.goto("http://127.0.0.1:8812/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000); return pg
        th = "document.documentElement.dataset.theme || 'dark'"
        pg = page("dark", "light")
        if pg.evaluate(th) != "light": bad.append("saved light mode not applied with empty localStorage")
        if pg.evaluate("__applied[0]") != "light:light": bad.append("apply not called: " + str(pg.evaluate("__applied")))
        pg = page("dark", "system", scheme="light")
        if pg.evaluate(th) != "dark": bad.append("system ignored the bridge (followed prefers-color-scheme)")
        pg.evaluate("__sys = 'light'; __decklightUiMode()")
        if pg.evaluate(th) != "light": bad.append("phone switch to light not followed")
        pg.evaluate("document.querySelector('[data-act=opensettings]').click()"); pg.wait_for_timeout(300)
        pg.evaluate("document.querySelector('[data-act=themeset][data-v=dark]').click()"); pg.wait_for_timeout(300)
        if pg.evaluate(th) != "dark" or pg.evaluate("localStorage.getItem('ygo-theme')") != "dark": bad.append("Dark button")
        if pg.evaluate("__applied.at(-1)") != "dark:dark": bad.append("apply after Dark: " + pg.evaluate("__applied.at(-1)"))
        pg.evaluate("__decklightUiMode()")
        if pg.evaluate(th) != "dark": bad.append("a phone switch changed an explicit Dark")
        b.close()
finally:
    srv.kill()
print("problems", bad); sys.exit(1 if bad else 0)
