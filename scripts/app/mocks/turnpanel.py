"""Mocks for the turn panel on a phone (user 2026-11-16, Watch screenshot: "It feels very compact and crowded";
"we can reduce the font size too ... it can be readable anyways"; "what will happen actually if the deck name or
username ... it's really big ... it doesn't fit regardless"). Watch at the user's phone (384 x 775, font scale 1.1) at
the same play, three ways:
  - now: "Blue-Eyes's turn" over the phase in the strip between the fields;
  - proposed: no names in the strip or the play feed. "Turn" carries the active player's mark (your dot, the
    opponent's seal, as in the LP bars and the log), the active LP bar is lit as now, names in the LP bars end in
    "..." when too long, and the board's large numbers are smaller on a phone (phase 17 -> 15 px, turn 24 -> 20,
    LP 24/20 -> 21/18);
  - proposed with two long names.
Games come from `writes_demo_games` (see games_check.py). Run from ygo-zero/:
    uv run --no-project --with playwright==1.57.0 --with pillow python -u scripts/app/mocks/turnpanel.py <dir>/records <out_dir>"""
import shutil, subprocess, sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path.cwd(); records = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
shutil.rmtree(out / "demo", ignore_errors=True); (out / "demo").mkdir()
shutil.copy(records.parent / "install_id", out / "demo" / "install_id")
records = Path(shutil.copytree(records, out / "demo" / "records", ignore=shutil.ignore_patterns("trash", "index.sqlite*")))
(records / "index.sqlite").touch()
# Android's font scale: every px font size x1.1 (as scripts/app/layout_check.py does).
ZOOM = """(k) => { for (const s of document.styleSheets) { let rs; try { rs = s.cssRules; } catch (e) { continue; }
  const walk = (rs) => { for (const r of rs) { if (r.cssRules) walk(r.cssRules); if (r.style && r.style.fontSize && r.style.fontSize.endsWith('px')) r.style.fontSize = (parseFloat(r.style.fontSize) * k) + 'px'; } };
  walk(rs); } }"""
CSS = """
/* the name leaves the strip (kept for screen readers) */
.phase .who { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.phase .pname { font-size: calc(15px * K); }
/* whose turn: the active player's mark before "Turn", the same mark as their LP bar and the log */
.turnno span { display: inline-flex; align-items: center; gap: 6px; }
.turnno span::before { content: ""; flex: none; width: 7px; height: 7px; border-radius: 50%; background: var(--you); }
#midstrip:has(.phase.bot) .turnno span::before { width: 9px; height: 9px; border-radius: 2px; background: none;
  box-shadow: inset 0 0 0 1.5px var(--bot), inset 0 0 0 3px transparent, inset 0 0 0 4.5px rgba(var(--bot-rgb), 0.55); }
.turnno b { font-size: calc(20px * K); }
.lp { font-size: calc(21px * K); }
.pbar.bot .lp { font-size: calc(18px * K); }
/* a long name ends in "..." and never pushes the bar's right side off the screen */
.lpblock { grid-template-columns: minmax(0, auto) auto; flex: 0 1 auto; }
.lpblock .pname { display: block; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.lpblock .pname::before { display: inline-block; margin-right: 7px; vertical-align: 0.1em; }
.pbar .right { flex: none; }
/* the play feed: the player's mark instead of their name ("BLUE-/EYES" wrapped in a 28 px column) */
.evs .who { flex: 0 0 9px; font-size: 0; align-self: center; width: 9px; height: 9px; }
.evs .ev.you .who { width: 7px; height: 7px; flex-basis: 7px; margin: 0 1px; border-radius: 50%; background: var(--you); }
.evs .ev.bot .who { border-radius: 2px; box-shadow: inset 0 0 0 1.5px var(--bot), inset 0 0 0 3px transparent, inset 0 0 0 4.5px rgba(var(--bot-rgb), 0.55); }
"""


def label(img_path, text):
    im = Image.open(img_path).convert("RGB")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    sheet = Image.new("RGB", (im.width, im.height + 64), "#0f0e0c")
    sheet.paste(im, (0, 64)); ImageDraw.Draw(sheet).text((20, 16), text, font=font, fill="#d6cfc0")
    return sheet


srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8846", "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(records)], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for opt in ("now", "new", "long"):
            pg = b.new_page(viewport={"width": 384, "height": 775}, device_scale_factor=2)
            pg.goto("http://127.0.0.1:8846/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
            pg.click('.lsub button[data-tab="games"]'); pg.wait_for_selector(".gm-row", timeout=30000)
            pg.locator(".gm-row:not(:has(.gm-s))").first.click(); pg.wait_for_selector('[data-act="games:watch"]', timeout=30000)
            pg.click('[data-act="games:watch"]'); pg.wait_for_selector("#watch .w-play", timeout=30000); pg.wait_for_timeout(800)
            if pg.locator('#watch [data-w="play"][aria-label="Pause"]').count(): pg.click('#watch [data-w="play"]')
            pg.frame_locator("#watch iframe").locator("#board").wait_for(timeout=30000)
            for _ in range(2):
                pg.click('#watch [data-w="next"]'); pg.wait_for_timeout(600)
            fr = pg.locator("#watch iframe").element_handle().content_frame()
            fr.evaluate(ZOOM, 1.1)
            if opt != "now":
                fr.evaluate("(css) => { const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s); }", CSS.replace("K", "1.1"))
            if opt == "long":
                fr.evaluate("""() => { const n = document.querySelectorAll('.lpblock .pname');
                  if (n[0]) n[0].lastChild.textContent = 'Dragonmaid Branded Despia (Labrynth variant)';
                  if (n[1]) n[1].lastChild.textContent = 'carlos_the_duelist_from_brazil'; }""")
            fr.evaluate("() => dispatchEvent(new Event('resize'))"); pg.wait_for_timeout(600)
            pg.screenshot(path=str(out / f"{opt}.png")); pg.close()
        b.close()
    ims = [label(out / "now.png", "now"), label(out / "new.png", "proposed"), label(out / "long.png", "proposed, long names")]
    sheet = Image.new("RGB", (sum(i.width for i in ims) + 40, ims[0].height), "#0f0e0c")
    x = 0
    for i in ims:
        sheet.paste(i, (x, 0)); x += i.width + 20
    sheet.save(out / "turnpanel.jpg", quality=84); print("sheet", out / "turnpanel.jpg", flush=True)
finally:
    srv.kill()
