"""Mocks for where "Watch the replay" sits on a game's page (user 2026-11-16: "it looks off ... it's just in the middle at
the top. Can you mock some other options"). The real game page with each option injected over it; dark and light.
Games come from `writes_demo_games` (see games_check.py). Run from ygo-zero/:
    uv run --no-project --with playwright==1.57.0 --with pillow python -u scripts/app/mocks/watch_button.py <dir>/records <out_dir>"""
import shutil, subprocess, sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path.cwd(); records = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
shutil.rmtree(out / "demo", ignore_errors=True); (out / "demo").mkdir()
shutil.copy(records.parent / "install_id", out / "demo" / "install_id")
records = Path(shutil.copytree(records, out / "demo" / "records", ignore=shutil.ignore_patterns("trash", "index.sqlite*")))
(records / "index.sqlite").touch()

CSS = """
.mk-foot { display: flex; align-items: center; gap: 12px; width: calc(100% + 32px); margin: 14px -16px -14px; padding: 0 16px;
  min-height: 52px; border-top: 1px solid var(--line); color: var(--accent); font-size: 15px; text-align: left; background: none; }
.mk-foot .ring { flex: none; box-sizing: border-box; min-width: 32px; min-height: 32px; max-width: 32px; max-height: 32px; padding: 0; display: grid; place-items: center; width: 32px; height: 32px; border-radius: 50%;
  border: 1px solid rgba(var(--accent-rgb), .55); background: var(--accent-tint); }
.mk-foot .ring svg { display: block; width: 15px; height: 15px; margin-left: 2px; color: var(--accent); }
.mk-foot .chev { margin-left: auto; width: 18px; height: 18px; color: var(--muted); }
.mk-wide { display: flex !important; width: 100%; justify-content: center; min-height: 48px; font-size: 15px; }
.mk-two { display: flex; gap: 8px; } .mk-two .ak-btn { flex: 1; min-height: 44px; }
.mk-tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0 6px; }
.mk-tiles button { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; min-height: 72px;
  border: 1px solid var(--line-2); border-radius: 12px; color: var(--text); font-size: 13px; background: none; }
.mk-tiles button svg { width: 20px; height: 20px; color: var(--muted); }
.mk-tiles button.pri { border-color: rgba(var(--accent-rgb), .55); background: var(--accent-tint); color: var(--accent); }
.mk-tiles button.pri svg { color: var(--accent); }
.mk-hplay { color: var(--accent) !important; } .mk-hplay svg { width: 22px; height: 22px; }
"""
SVG = lambda d: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + d + "</svg>"  # noqa: E731
PLAY, CHECK, ASK = SVG('<path d="M8 5.5v13l10-6.5z"/>'), SVG('<path d="M5 12.5l4.5 4.5L19 7.5"/>'), SVG('<path d="M5 6h14v9H10l-4 3.5V15H5z"/>')
CHEV = SVG('<path d="M9.5 6l6 6-6 6"/>').replace("<svg ", '<svg class="chev" ')
JS = """([opt, css, PLAY, CHECK, ASK, CHEV]) => {
  const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s);
  const watch = document.querySelector('[data-act="games:watch"]'), wrow = watch.closest('.gm-acts');
  const card = document.querySelector('.ab-rb'), check = document.querySelector('[data-act="games:check"]');
  const crow = check && check.closest('.gm-acts');
  if (opt === 'A') {            // in the game card, as its footer
    wrow.remove();
    card.insertAdjacentHTML('beforeend', '<button class="mk-foot"><span class="ring">' + PLAY + '</span>Watch the replay' + CHEV + '</button>');
  } else if (opt === 'B') {     // one wide button under the card, the two tools as a pair below
    card.after(wrow); watch.classList.add('mk-wide'); crow && crow.classList.add('mk-two');
  } else if (opt === 'C') {     // a play icon in the header, beside Export
    wrow.remove();
    document.querySelector('.gm-xp').insertAdjacentHTML('beforebegin', '<button class="ak-ibtn mk-hplay" aria-label="Watch the replay">' + PLAY + '</button>');
    document.querySelector('.gm-xp').style.marginLeft = '0';
  } else if (opt === 'D') {     // three equal actions under the card
    wrow.remove(); crow && crow.remove();
    card.insertAdjacentHTML('afterend', '<div class="mk-tiles"><button class="pri">' + PLAY + 'Watch</button><button>' + CHECK +
      'Check decisions</button><button>' + ASK + 'Ask</button></div>');
  }
}"""
OPTS = {"0": "Now", "A": "A · Footer of the game card", "B": "B · Wide, under the card", "C": "C · Icon in the header", "D": "D · Three equal actions"}
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8834", "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(records)], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for theme in ("dark", "light"):
            for opt in OPTS:
                pg = b.new_page(viewport={"width": 412, "height": 820}, device_scale_factor=2)
                pg.add_init_script(f"try {{ localStorage.setItem('ygo-theme', '{theme}'); }} catch (e) {{}}")
                pg.goto("http://127.0.0.1:8834/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
                pg.click('.lsub button[data-tab="games"]'); pg.wait_for_selector(".gm-row", timeout=30000)
                pg.locator(".gm-row:not(:has(.gm-s))").first.click(); pg.wait_for_selector('[data-act="games:watch"]', timeout=30000)
                if opt != "0": pg.evaluate(JS, [opt, CSS, PLAY, CHECK, ASK, CHEV])
                pg.wait_for_timeout(300); pg.screenshot(path=str(out / f"{theme}-{opt}.png"))
                pg.close()
        b.close()
finally:
    srv.kill()
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
for theme in ("dark", "light"):
    ims = [Image.open(out / f"{theme}-{o}.png") for o in OPTS]
    w, h = ims[0].size; pad = 40
    sheet = Image.new("RGB", (len(ims) * (w + pad) + pad, h + 110), "#0f0e0c" if theme == "dark" else "#e9e4d8")
    d = ImageDraw.Draw(sheet)
    for i, (o, im) in enumerate(zip(OPTS.values(), ims)):
        x = pad + i * (w + pad); sheet.paste(im, (x, 90))
        d.text((x, 30), o, font=font, fill="#d6cfc0" if theme == "dark" else "#2a2721")
    sheet.save(out / f"watch-button-{theme}.jpg", quality=85)
    print("sheet", out / f"watch-button-{theme}.jpg", flush=True)
