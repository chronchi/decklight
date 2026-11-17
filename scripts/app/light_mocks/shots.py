"""Light theme mocks (2026-11-15): the app with palettes A Tufte, B Solarized, C Washi injected over the dark CSS.
Run from ygo-zero/: uv run --no-project --with playwright==1.57.0 python -u scripts/app/light_mocks/shots.py <out_dir>
(The mock CSS predates the tokens in web/styles.css; kept as the record of what was shown.)"""
import subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(Path(__file__).parent)); from pal import P
ROOT = Path.cwd(); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
def rgba(h, a): return f"rgba({int(h[1:3],16)}, {int(h[3:5],16)}, {int(h[5:7],16)}, {a})"
def css(p):
    return (":root{" + f"--bg:{p['bg']};--felt:{p['felt']};--zone:{p['zone']};--sheet:{p['sheet']};--text:{p['text']};--muted:{p['muted']};"
            f"--faint:{p['faint']};--you:{p['you']};--bot:{p['bot']};--accent:{p['accent']};--danger:{p['danger']};--bot-ink:{p['danger']};--neg:{p['neg']};"
            f"--zone-line:{rgba(p['text'],.10)};--line:{rgba(p['text'],.08)};--line-2:{rgba(p['text'],.16)};--raise:{rgba(p['text'],.035)};"
            f"--accent-tint:{rgba(p['accent'],.10)};--indigo-1:{p['i1']};--indigo-2:{p['i2']};--art:none;color-scheme:light;}}"
            f"body{{background:{p['bg']};color:{p['text']}}}"
            f"#app{{background:radial-gradient(120% 60% at 50% 42%, {p['sheet']} 0%, var(--felt) 55%, var(--bg) 100%)}}"
            # Card backs are the physical card: they stay deep indigo in every theme.
            ".handcount i, .card.back, .thumb.back, .cue-card.back{--indigo-1:#22304a;--indigo-2:#161e2e}")
INJ = "(c) => { let s = document.getElementById('lt'); if (!s) { s = document.createElement('style'); s.id = 'lt'; document.head.appendChild(s); } s.textContent = c; }"
import shutil
def main():
  for name, p in [("0-dark", None)] + list(P.items()):
      if (out / f"{name}-4duel.png").exists(): continue
      shutil.rmtree(out / "ask", ignore_errors=True); shutil.rmtree(out / "records", ignore_errors=True)
      srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8815", "--token", "smoke", "--ask-dir", str(out / "ask"),
                          "--records", str(out / "records")], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
      try:
        time.sleep(3)
        with sync_playwright() as pw:
          b = pw.chromium.launch()
          if True:
              pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
              pg.add_init_script("document.addEventListener('DOMContentLoaded', () => {})")
              pg.goto("http://127.0.0.1:8815/?t=smoke&ltab=play"); pg.wait_for_selector(".bbar svg", timeout=60000)
              if p: pg.evaluate(INJ, css(p))
              pg.wait_for_timeout(1500); pg.screenshot(path=str(out / f"{name}-1play.png"))
              pg.click('[data-tab="decks"]'); pg.wait_for_timeout(1200); pg.screenshot(path=str(out / f"{name}-2decks.png"))
              pg.click('[data-tab="ask"]'); pg.wait_for_timeout(800); pg.click('[data-act="ask:model"]'); pg.wait_for_timeout(1200)
              pg.screenshot(path=str(out / f"{name}-3ask.png")); pg.keyboard.press("Escape")
              pg.click('[data-tab="play"]'); pg.wait_for_selector('#startbtn:not([disabled])', timeout=60000); pg.click("#startbtn")
              for _ in range(12):
                  if pg.locator('[data-act="gamemenu"]').count(): break
                  for sel in ('[data-act="rpspick"]', '[data-act="rpsagain"]', '[data-act="rpsfirst"]', '[data-act="rpsbotfirst"]'):
                      if pg.locator(sel).count(): pg.locator(sel).first.click(); break
                  pg.wait_for_timeout(1500)
              pg.wait_for_timeout(4000); pg.screenshot(path=str(out / f"{name}-4duel.png"))
              pg.close(); print("done", name, flush=True)
          b.close()
      finally:
        srv.kill(); srv.wait()

if __name__ == "__main__":
    main()
