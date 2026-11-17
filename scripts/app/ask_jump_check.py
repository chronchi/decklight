"""Ask's jump-to-latest button reaches the end of a long thread (2026-11-15: "it never goes to the bottom"). The thread
is padded with tall filler, scrolled to the top, and the button is tapped; it must end within 24 px of the end.
    uv run --no-project --with playwright==1.57.0 python -u scripts/app/ask_jump_check.py <out_dir>"""
import subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path.cwd(); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8818", "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(out / "records")], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
SC = "#lobby.askview .lscroll"
FILL = """(sc) => { const t = document.querySelector(sc);
  for (let i = 0; i < 12; i++) { const d = document.createElement('div'); d.className = 'filler'; d.style.height = '600px'; d.style.flex = 'none';
    d.textContent = 'filler ' + i; t.appendChild(d); }
  return [t.scrollHeight, t.clientHeight, getComputedStyle(t).overflowY]; }"""
LEFT = "(sc) => { const t = document.querySelector(sc); return t.scrollHeight - t.scrollTop - t.clientHeight; }"
problems = []
try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 412, "height": 915}, has_touch=True)
        pg.on("pageerror", lambda e: problems.append(str(e)))
        pg.goto("http://127.0.0.1:8818/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
        pg.click('[data-tab="ask"]'); pg.wait_for_selector(".ak-empty"); pg.wait_for_timeout(2000)
        print("filled", pg.evaluate(FILL, SC), flush=True)
        pg.evaluate("(sc) => { const t = document.querySelector(sc); t.scrollTop = t.scrollHeight; }", SC); pg.wait_for_timeout(300)
        pg.evaluate("(sc) => { document.querySelector(sc).scrollTop = 0; }", SC); pg.wait_for_timeout(500)
        print("at top: px left", pg.evaluate(LEFT, SC), "fillers", pg.locator(".filler").count(), flush=True)
        if not pg.locator(".ak-jump:visible").count():
            problems.append("no jump button after scrolling to the top")
        else:
            pg.tap(".ak-jump"); pg.wait_for_timeout(2500)
            left = pg.evaluate(LEFT, SC)
            print("px left after the jump:", left, flush=True)
            if left > 24: problems.append(f"the jump stopped {left} px before the end")
            if pg.locator(".ak-jump:visible").count(): problems.append("the button still shows at the end")
        b.close()
finally:
    srv.kill()
print("problems", problems)
sys.exit(1 if problems else 0)
