"""Mocks for a larger board in Watch (user 2026-11-16: "Why is the replay bar smaller than the actual board game when I'm
playing the game?"; then, on the two ideas: "Yes, mock both, I like these ideas"). Watch on the user's phone (384 x 775,
font scale 1.1) at the same play:
  - now: a header (title, Export, close) over the board and the controls under it;
  - A, one bar: no header; close and Export join the controls row (close at the far left, Export at the far right);
  - B, controls hide while playing: the board takes the whole screen; while it plays only a thin progress line shows,
    and a tap (or a pause) brings the header and the controls back over the board.
The user chose "The combination" (2026-11-16): A's one bar, hidden while playing as in B (built in watch.js).
Games come from `writes_demo_games` (see games_check.py). Run from ygo-zero/:
    uv run --no-project --with playwright==1.57.0 --with pillow python -u scripts/app/mocks/watchbar.py <dir>/records <out_dir>"""
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
  walk(rs); dispatchEvent(new Event('resize')); } }"""
ONE_BAR = """() => {
  const s = document.createElement('style');
  s.textContent = `#watch .w-top { display: none; }
    #watch .w-row .w-x, #watch .w-row .w-xp { color: var(--muted); }
    #watch .w-row .w-xp span { display: none; }`;
  document.head.appendChild(s);
  const top = document.querySelector('#watch .w-top'), row = document.querySelector('#watch .w-row');
  const cl = top.querySelector('[data-w="close"]'), ex = top.querySelector('[data-w="export"]');
  cl.classList.add('w-x'); ex.classList.add('w-xp'); ex.classList.remove('w-pill');
  row.prepend(cl); row.append(ex);
}"""
OVERLAY = """(playing) => {
  const s = document.createElement('style');
  s.textContent = `#watch .w-stage { position: absolute; inset: 0; }
    #watch .w-top, #watch .w-bar { position: absolute; left: 0; right: 0; z-index: 3; transition: opacity 200ms;
      background: var(--bg); box-shadow: 0 0 24px 8px var(--bg); }
    #watch .w-top { top: 0; } #watch .w-bar { bottom: 0; }
    #watch.hidebar .w-top, #watch.hidebar .w-bar { opacity: 0; pointer-events: none; }
    #watch .w-line { position: absolute; left: 0; right: 0; bottom: 0; height: 2px; z-index: 2; background: var(--line-2); }
    #watch .w-line > i { position: absolute; inset: 0 auto 0 0; background: var(--accent); opacity: .75; }`;
  document.head.appendChild(s);
  const w = document.getElementById('watch');
  if (playing) {
    w.classList.add('hidebar');
    const p = document.querySelector('#watch .w-prog > i');
    const line = document.createElement('div'); line.className = 'w-line';
    line.innerHTML = '<i style="width:' + (p ? p.style.width : '20%') + '"></i>';
    w.appendChild(line);
  }
}"""


def label(img_path, text):
    im = Image.open(img_path).convert("RGB")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    sheet = Image.new("RGB", (im.width, im.height + 64), "#0f0e0c")
    sheet.paste(im, (0, 64)); ImageDraw.Draw(sheet).text((20, 16), text, font=font, fill="#d6cfc0")
    return sheet


srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8849", "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(records)], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
OPTS = [("now", "now"), ("a", "A · one bar"), ("b-play", "B · while playing"), ("b-paused", "B · paused or tapped")]
try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for opt, _ in OPTS:
            pg = b.new_page(viewport={"width": 384, "height": 775}, device_scale_factor=2)
            pg.goto("http://127.0.0.1:8849/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
            pg.click('.lsub button[data-tab="games"]'); pg.wait_for_selector(".gm-row", timeout=30000)
            pg.locator(".gm-row:not(:has(.gm-s))").first.click(); pg.wait_for_selector('[data-act="games:watch"]', timeout=30000)
            pg.click('[data-act="games:watch"]'); pg.wait_for_selector("#watch .w-play", timeout=30000); pg.wait_for_timeout(800)
            if pg.locator('#watch [data-w="play"][aria-label="Pause"]').count(): pg.click('#watch [data-w="play"]')
            pg.frame_locator("#watch iframe").locator("#board").wait_for(timeout=30000)
            for _ in range(5):
                pg.click('#watch [data-w="next"]'); pg.wait_for_timeout(700)
            pg.wait_for_timeout(1500)
            fr = pg.locator("#watch iframe").element_handle().content_frame()
            fr.evaluate(ZOOM, 1.1)
            if opt == "a": pg.evaluate(ONE_BAR)
            if opt.startswith("b"): pg.evaluate(OVERLAY, opt == "b-play")
            pg.wait_for_timeout(900)
            print(opt, "card", fr.evaluate("() => getComputedStyle(document.documentElement).getPropertyValue('--card-w')"), flush=True)
            pg.screenshot(path=str(out / f"{opt}.png")); pg.close()
        b.close()
    ims = [label(out / f"{o}.png", t) for o, t in OPTS]
    sheet = Image.new("RGB", (sum(i.width for i in ims) + 20 * (len(ims) - 1), ims[0].height), "#0f0e0c")
    x = 0
    for i in ims:
        sheet.paste(i, (x, 0)); x += i.width + 20
    sheet.save(out / "watchbar.jpg", quality=82); print("sheet", out / "watchbar.jpg", flush=True)
finally:
    srv.kill()
