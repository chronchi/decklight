"""Mocks for the table on wide screens (user 2026-11-16: "there's still a lot of black screens, black stripes on the side
... it's just weird that it looks empty in general ... Check EDO Pro as well"). A saved game in Watch at a mid-game turn,
as it is now and with the proposal injected into the play screen:
  - three columns across the window: the card reader (left), the table, the log (right); the reader is never empty
    (it shows the card of the last play until you tap another);
  - the felt fills the window with a quiet seigaiha pattern, and each player's half sits on its own mat in their colour;
  - every empty zone carries a small drawn mark of what it holds (no words), like EDOPro's zone icons but quieter.
Games come from `writes_demo_games` (see games_check.py). Run from ygo-zero/:
    uv run --no-project --with playwright==1.57.0 --with pillow python -u scripts/app/mocks/table.py <dir>/records <out_dir>"""
import shutil, subprocess, sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path.cwd(); records = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
shutil.rmtree(out / "demo", ignore_errors=True); (out / "demo").mkdir()
shutil.copy(records.parent / "install_id", out / "demo" / "install_id")
records = Path(shutil.copytree(records, out / "demo" / "records", ignore=shutil.ignore_patterns("trash", "index.sqlite*")))
(records / "index.sqlite").touch()
DEVICES = {"16in": (1728, 1035), "13in": (1280, 720), "tab-l": (1180, 820)}
# Seigaiha (overlapping waves), drawn in the accent at a whisper: a texture, never a picture.
WAVES = ("data:image/svg+xml;utf8," +
         "<svg xmlns='http://www.w3.org/2000/svg' width='48' height='24' viewBox='0 0 48 24' fill='none' stroke='%23c8a96a' stroke-opacity='0.07' stroke-width='1'>"
         "<path d='M0 24a24 24 0 0 1 48 0M8 24a16 16 0 0 1 32 0M16 24a8 8 0 0 1 16 0'/>"
         "<path d='M-24 12a24 24 0 0 1 48 0M-16 12a16 16 0 0 1 32 0M-8 12a8 8 0 0 1 16 0'/>"
         "<path d='M24 12a24 24 0 0 1 48 0M32 12a16 16 0 0 1 32 0M40 12a8 8 0 0 1 16 0'/></svg>")
CSS = """
#layout { display: grid !important; grid-template-columns: minmax(280px, 380px) minmax(0, 1fr) minmax(300px, 380px); height: 100dvh;
  background: radial-gradient(70% 80% at 50% 48%, var(--glow) 0%, transparent 70%), url("WAVES"), var(--felt) !important; }
#readside { grid-column: 1; grid-row: 1; min-height: 0; overflow-y: auto; padding: 22px 22px 16px;
  border-right: 1px solid var(--line); background: color-mix(in srgb, var(--bg) 78%, transparent); }
#readside .sidecard { padding: 0; border: 0; flex: none; }
#readside .dface { position: relative; width: min(100%, 250px); aspect-ratio: 59 / 86; margin: 0 auto 16px; overflow: hidden;
  border-radius: 8px; box-shadow: 0 10px 30px rgba(var(--shade-rgb), 0.5); }
#readside .dface > * { position: absolute !important; inset: 0 !important; width: 100% !important; height: 100% !important; object-fit: cover; }
#readside h2 { font-size: 18px; font-weight: 500; margin: 0 0 4px; }
#readside .typeline { font-size: 12.5px; color: var(--muted); margin-bottom: 10px; }
#readside .dtext { font-size: 13.5px; line-height: 1.55; color: var(--text); }
#readside .rlabel { font-variant-caps: all-small-caps; letter-spacing: 0.14em; font-size: 12px; color: var(--muted); margin: 0 0 12px; }
#app { grid-column: 2; grid-row: 1; justify-self: center; background: transparent !important; border: 0 !important; width: var(--app-w); }
#logside { grid-column: 3; grid-row: 1; width: auto !important; flex: none !important; border-right: 0 !important;
  background: color-mix(in srgb, var(--bg) 78%, transparent) !important; }
.field { border-radius: 14px; padding: 5px 0; }
.field.you { background: rgba(var(--you-rgb), 0.045); box-shadow: inset 0 0 0 1px rgba(var(--you-rgb), 0.16); }
.field.bot { background: rgba(var(--bot-rgb), 0.04); box-shadow: inset 0 0 0 1px rgba(var(--bot-rgb), 0.14); }
/* Each empty zone carries a small drawn mark (no words): a ring for a monster, a lozenge for a spell or trap, a ring
   with a centre for an Extra Monster Zone. Decoration only: the zones keep their labels for screen readers. */
.zrow .slot.empty:not(.side):not(.fieldz)::after, #midstrip .emz.empty::after {
  content: ""; position: absolute; left: 50%; top: 50%; width: 30%; aspect-ratio: 1; transform: translate(-50%, -50%);
  background: var(--muted); opacity: 0.32; -webkit-mask: var(--g) center / contain no-repeat; mask: var(--g) center / contain no-repeat; }
[data-row="mzone"] .slot.empty { --g: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='8'/></svg>"); }
[data-row="szone"] .slot.empty { --g: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 4l7 8-7 8-7-8z'/></svg>"); }
#midstrip .emz.empty { --g: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='8'/><circle cx='12' cy='12' r='2.2'/></svg>"); }
"""
JS = """(css) => {
  const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s);
  const side = document.querySelector('#logside .sidecard');
  const col = document.createElement('div'); col.id = 'readside';
  col.innerHTML = '<p class="rlabel">Card</p>';
  document.getElementById('layout').prepend(col);
  if (side) col.appendChild(side);
  // The reader: the zoom viewer's own parts (art, name, type line, text), as the column would draw them.
  const d = document.getElementById('detail');
  if (d && side) {
    side.innerHTML = '';
    for (const k of ['.dface', 'h2', '.typeline', '.dtext']) { const e = d.querySelector(k); if (e) side.appendChild(e.cloneNode(true)); }
    d.remove();
  }
}"""


def label(img_path, text):
    im = Image.open(img_path).convert("RGB")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    sheet = Image.new("RGB", (im.width, im.height + 60), "#0f0e0c")
    sheet.paste(im, (0, 60)); ImageDraw.Draw(sheet).text((20, 16), text, font=font, fill="#d6cfc0")
    return sheet


srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8840", "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(records)], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for dev, (w, h) in DEVICES.items():
            for opt in ("now", "new"):
                pg = b.new_page(viewport={"width": w, "height": h})
                pg.goto("http://127.0.0.1:8840/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
                pg.click('.lsub button[data-tab="games"]'); pg.wait_for_selector(".gm-row", timeout=30000)
                pg.locator(".gm-row:not(:has(.gm-s))").nth(1).click(); pg.wait_for_selector('[data-act="games:watch"]', timeout=30000)
                pg.click('[data-act="games:watch"]'); pg.wait_for_selector("#watch .w-play", timeout=30000); pg.wait_for_timeout(800)
                if pg.locator('#watch [data-w="play"][aria-label="Pause"]').count(): pg.click('#watch [data-w="play"]')
                for _ in range(16):
                    pg.click('#watch [data-w="last"]'); pg.wait_for_timeout(200)
                pg.wait_for_timeout(1200)
                frame = pg.locator("#watch iframe").element_handle().content_frame()
                if opt == "new":
                    # The reader shows a card on the table (the proposal: the last play's card until you tap another).
                    frame.locator('.field [data-row="mzone"] .slot:not(.empty):not(.side)').first.click(); frame.wait_for_timeout(500)
                    frame.evaluate(JS, CSS.replace("WAVES", WAVES)); frame.wait_for_timeout(600)
                pg.screenshot(path=str(out / f"{dev}-{opt}.png")); pg.close()
            a, c = label(out / f"{dev}-now.png", f"{dev} · now"), label(out / f"{dev}-new.png", f"{dev} · proposed")
            sheet = Image.new("RGB", (a.width, a.height + c.height + 20), "#0f0e0c"); sheet.paste(a, (0, 0)); sheet.paste(c, (0, a.height + 20))
            sheet.save(out / f"table-{dev}.jpg", quality=84); print("sheet", out / f"table-{dev}.jpg", flush=True)
        b.close()
finally:
    srv.kill()
