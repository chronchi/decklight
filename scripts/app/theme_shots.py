"""Screenshots of the main screens in each theme (Settings → Appearance), for review and for a before/after diff of the
dark theme. Run from ygo-zero/ with ORT_DYLIB_PATH set for the duel (see sheets_check.py):
    uv run --no-project --with playwright==1.57.0 python -u scripts/app/theme_shots.py <out_dir> [dark light system-light]
Writes <out_dir>/<theme>-<screen>.png. WEB_DIR=<dir> serves another copy of web/ (a baseline from `git archive`). Exits 1 on a page error or when the page's theme is not the one asked for."""
import os, shutil, subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path.cwd(); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
themes = sys.argv[2:] or ["dark", "light"]
problems = []
for theme in themes:
    # system-light: the System choice with the browser reporting a light scheme.
    mode, scheme = ("system", "light") if theme == "system-light" else (theme, "dark")
    shutil.rmtree(out / "ask", ignore_errors=True); shutil.rmtree(out / "records", ignore_errors=True)
    srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8814", "--token", "smoke",
                            "--ask-dir", str(out / "ask"), "--records", str(out / "records")]
                           + (["--web-dir", os.environ["WEB_DIR"]] if os.environ.get("WEB_DIR") else []),
                           stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
    try:
        time.sleep(3)
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2, color_scheme=scheme)
            pg.on("pageerror", lambda e: problems.append(f"{theme}: page error: {e}"))
            pg.add_init_script(f"try {{ localStorage.setItem('ygo-theme', '{mode}') }} catch (e) {{}}")
            pg.goto("http://127.0.0.1:8814/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
            got = pg.evaluate("document.documentElement.dataset.theme || 'dark'")
            want = "light" if theme in ("light", "system-light") else "dark"
            if got != want: problems.append(f"{theme}: the page is {got}, not {want}")
            def shot(name): pg.wait_for_timeout(900); pg.screenshot(path=str(out / f"{theme}-{name}.png"))
            shot("1-lobby")
            pg.click('[data-act="opensettings"]'); shot("2-settings")
            pg.keyboard.press("Escape"); pg.wait_for_timeout(400)
            pg.click('[data-tab="decks"]'); shot("3-decks")
            pg.click('[data-tab="ask"]'); pg.wait_for_timeout(500); pg.click('[data-act="ask:model"]'); shot("4-ask")
            pg.keyboard.press("Escape")
            pg.click('[data-tab="play"]'); pg.wait_for_selector('#startbtn:not([disabled])', timeout=60000); pg.click("#startbtn")
            for _ in range(12):
                if pg.locator('[data-act="gamemenu"]').count(): break
                for sel in ('[data-act="rpspick"]', '[data-act="rpsagain"]', '[data-act="rpsfirst"]', '[data-act="rpsbotfirst"]'):
                    if pg.locator(sel).count(): pg.locator(sel).first.click(); break
                pg.wait_for_timeout(1500)
            pg.wait_for_timeout(4000); shot("5-duel")
            info = pg.locator("#sheet .infobtn").first   # a card's details over the duel
            if info.count(): info.click(timeout=5000); shot("6-card")
            b.close(); print("done", theme, flush=True)
    finally:
        srv.kill(); srv.wait()
print("problems", problems)
sys.exit(1 if problems else 0)
