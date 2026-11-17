"""Layout mocks for every screen size (user 2026-11-16: "the app should be able to automatically resize to whatever screen
we're using ... consistent across devices ... if I'm using the laptop, I want to have the laptop experience that is good";
then "it might be used in a tablet too ... the folds from Samsung ... things that can be reused across ... devices"). Every pane
is the real app rendered at the pane's size (its own header, sub-tabs and bottom bar hidden, since the frame navigates),
placed in a frame drawn here:
  A  a left rail (mark, Play · Puzzles · Decks · Ask, settings at the foot), panes side by side;
  B  a top bar (mark and wordmark, the same four sections, settings), the same panes under it.
The play surfaces (a duel, Watch) scale with the window instead of staying phone-sized. Screens: Play (setup beside your
games), a saved game (list · game · its replay), Ask beside the game it is about, a duel. Games come from
`writes_demo_games` (see games_check.py). Run from ygo-zero/ with ORT_DYLIB_PATH set:
    uv run --no-project --with playwright==1.57.0 python -u scripts/app/mocks/laptop.py <dir>/records <out_dir>"""
import base64, shutil, subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw, ImageFont

ROOT = Path.cwd(); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); rec0 = Path(sys.argv[1])
shutil.rmtree(out / "demo", ignore_errors=True); (out / "demo").mkdir()
shutil.copy(rec0.parent / "install_id", out / "demo" / "install_id")
rec = Path(shutil.copytree(rec0, out / "demo" / "records", ignore=shutil.ignore_patterns("trash", "index.sqlite*"))); (rec / "index.sqlite").touch()
PORT = 8836; BASE = f"http://127.0.0.1:{PORT}"
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", str(PORT), "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(rec)], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
# Laptop screens as browser viewports (window minus the browser's bars).
# Devices by the space the app gets (css px): a phone; a foldable unfolded (Galaxy Z Fold inner screen); an 11" tablet
# both ways; a 13" and a 16" laptop browser (window minus the browser's bars). The layout follows the width class, not
# the device: compact < 600 (bottom bar, one pane), medium 600-839 (rail, one pane), expanded 840-1599 (rail, two
# panes), large >= 1600 (rail, three panes). A duel and Watch take the whole screen and grow with it.
DEVICES = [("phone", "Phone", 412, 915), ("fold", "Foldable, open", 773, 900), ("tab-p", "Tablet, portrait", 820, 1180),
           ("tab-l", "Tablet, landscape", 1180, 820), ("13in", '13" laptop', 1280, 720), ("16in", '16" laptop', 1728, 1035)]


def wclass(w):
    return "compact" if w < 600 else "medium" if w < 840 else "expanded" if w < 1600 else "large"
RAIL, TOP = 84, 60
# Inside a pane the frame does the navigation; content columns take the pane's width.
PANE = """.lhead, .lsub, .bbar { display: none !important; }
#app { max-width: none !important; } #lobby .lscroll { padding-top: 22px; }
.lscroll > * { max-width: 760px; width: 100%; box-sizing: border-box; margin-left: auto; margin-right: auto; }
.ak-composer { padding-left: max(18px, calc((100% - 760px) / 2)); padding-right: max(18px, calc((100% - 760px) / 2)); }
#watch .w-bar { align-self: center; width: min(100%, 820px); box-sizing: border-box; }
#lobby:not(:has(#startwrap)) ~ #logside, body:not(:has(#board)) #logside { display: none !important; }
body:not(:has(#board)) #app { flex: 1 1 auto !important; border: 0 !important; }
#lobby:has(#startwrap) { justify-content: flex-start; } #lobby:has(#startwrap) .lscroll { flex: 0 1 auto; }
#startwrap { max-width: 760px; width: 100%; margin: 0 auto; box-sizing: border-box; }
/* The table: the felt runs to the window's edges, so a wide window is a calm table, not a phone on a desk. */
body:has(#board) { background: radial-gradient(90% 70% at 50% 45%, var(--glow) 0%, var(--felt) 60%, var(--bg) 100%); }
body:has(#board) #app { background: transparent; border-left: 0 !important; }
body:has(#board) #logside { background: rgba(var(--panel-rgb), .55); }"""
PLAY_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 5.5v13l10-6.5z"/></svg>'


def page(b, w, h, z=1.0):
    pg = b.new_page(viewport={"width": round(w / z), "height": round(h / z)}, device_scale_factor=z)
    pg.goto(BASE + "/?t=smoke"); pg.wait_for_selector(".bbar svg, [data-act=gamemenu]", timeout=60000); pg.wait_for_timeout(800)
    return pg


def shot(pg, name):
    pg.add_style_tag(content=PANE); pg.wait_for_timeout(700)
    pg.screenshot(path=str(out / name)); pg.close()
    return name


def games_list(pg):
    pg.click('.lsub button[data-tab="games"]'); pg.wait_for_selector(".gm-row", timeout=30000)


def game(pg):
    games_list(pg); pg.locator(".gm-row:not(:has(.gm-s))").first.click(); pg.wait_for_selector(".gm-tiles", timeout=30000)


def watch(pg):
    game(pg); pg.click('[data-act="games:watch"]'); pg.wait_for_selector("#watch .w-play", timeout=30000); pg.wait_for_timeout(800)
    if pg.locator('#watch [data-w="play"][aria-label="Pause"]').count(): pg.click('#watch [data-w="play"]')
    for _ in range(4): pg.click('#watch [data-w="last"]'); pg.wait_for_timeout(400)
    for _ in range(3): pg.click('#watch [data-w="next"]'); pg.wait_for_timeout(400)
    pg.wait_for_timeout(1200)


def setup(pg):
    pg.click('.lsub button[data-tab="play"]'); pg.wait_for_selector("#startbtn:not([disabled])", timeout=60000)


def ask_game(pg):
    game(pg); pg.click('[data-act="games:askgame"]'); pg.wait_for_selector("#akin", timeout=30000); pg.wait_for_timeout(800)


def duel(pg):
    if not pg.locator('[data-act="gamemenu"]').count():
        setup(pg); pg.click("#startbtn")
        for _ in range(12):
            if pg.locator('[data-act="gamemenu"]').count(): break
            for sel in ('[data-act="rpspick"]', '[data-act="rpsagain"]', '[data-act="rpsfirst"]', '[data-act="rpsbotfirst"]'):
                if pg.locator(sel).count(): pg.locator(sel).first.click(); break
            pg.wait_for_timeout(1500)
        pg.wait_for_selector('[data-act="gamemenu"]', timeout=60000)
    pg.wait_for_timeout(2500)


def play_zoom(w, h):
    # The play surfaces grow with the screen. Landscape and wide: the board column with the side panel (~880 x 860 css px).
    # Portrait: the phone layout (~520 x 880) scaled up.
    if w >= 900 and w > h:
        return max(1.0, min(w / 1000, h / 860, 1.5))
    return max(1.0, min(w / 520, h / 880, 2.0))


def frame_html(kind, W, H, nav, sel, gear, mark, panes):
    def img(name):
        return "data:image/png;base64," + base64.b64encode((out / name).read_bytes()).decode()
    items = "".join(f'<a class="{"sel" if k == sel else ""}">{ico}<span>{lab}</span></a>' for k, lab, ico in nav)
    cells = "".join(f'<div class="pane" style="width:{w}px"><img src="{img(n)}" width="{w}" height="{h}"></div>' for n, w, h in panes)
    if kind == "none":
        body = f'<div class="wrap">{cells}</div>'
    elif kind == "A":
        chrome = f'<nav class="rail"><div class="rmark">{mark}</div>{items}<div class="grow"></div><a class="gear">{gear}</a></nav>'
        body = f'<div class="wrap A">{chrome}<div class="main">{cells}</div></div>'
    else:
        chrome = f'<header class="top"><div class="tmark">{mark}<b>decklight</b></div><nav class="tnav">{items}</nav><a class="gear">{gear}</a></header>'
        body = f'<div class="wrap B">{chrome}<div class="main">{cells}</div></div>'
    return f"""<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="/styles.css"><style>
html, body {{ margin: 0; width: {W}px; height: {H}px; overflow: hidden; background: var(--bg); color: var(--text); }}
.wrap {{ display: flex; width: {W}px; height: {H}px; }} .wrap.B {{ flex-direction: column; }}
.main {{ display: flex; flex: 1; min-width: 0; min-height: 0; }}
.pane {{ flex: none; border-right: 1px solid var(--line); overflow: hidden; }} .pane:last-child {{ border-right: 0; }}
.pane img {{ display: block; }}
nav a {{ display: flex; align-items: center; color: var(--muted); font-size: 12px; position: relative; }}
nav a svg {{ width: 24px; height: 24px; }} nav a.sel {{ color: var(--text); }} nav a.sel .node {{ fill: var(--accent); }}
.rail {{ flex: 0 0 {RAIL}px; display: flex; flex-direction: column; align-items: stretch; gap: 6px; padding: 18px 0 14px;
  border-right: 1px solid var(--line-2); background: var(--sheet); --icon-fill: var(--sheet); }}
.rail .rmark {{ display: flex; justify-content: center; margin-bottom: 18px; }} .rail .rmark svg {{ height: 30px; width: auto; overflow: visible; }}
.rail a {{ flex-direction: column; justify-content: center; gap: 4px; min-height: 64px; }}
.rail a.sel::before {{ content: ""; position: absolute; left: 0; top: 18%; bottom: 18%; width: 2px; background: var(--accent); border-radius: 0 2px 2px 0; }}
.grow {{ flex: 1; }} .gear {{ justify-content: center; min-height: 48px; color: var(--muted); }} .gear svg {{ width: 22px; height: 22px; }}
.top {{ flex: 0 0 {TOP}px; display: flex; align-items: center; padding: 0 22px; border-bottom: 1px solid var(--line-2); background: var(--sheet); --icon-fill: var(--sheet); }}
.tmark {{ display: flex; align-items: center; gap: 10px; flex: 1; }} .tmark svg {{ height: 26px; width: auto; overflow: visible; }}
.tmark b {{ font-weight: 300; font-size: 18px; letter-spacing: .08em; }}
.tnav {{ display: flex; gap: 6px; }} .tnav a {{ flex-direction: row; gap: 8px; padding: 0 16px; height: {TOP}px; font-size: 14px; }}
.tnav a svg {{ width: 20px; height: 20px; }}
.tnav a.sel::after {{ content: ""; position: absolute; left: 16px; right: 16px; bottom: 0; height: 2px; background: var(--accent); }}
.top .gear {{ flex: 1; justify-content: flex-end; }}
</style></head><body>{body}</body></html>"""


try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = page(b, 412, 900)
        nav = pg.evaluate("""() => [...document.querySelectorAll('.bbar button')].map(x => [x.querySelector('span').textContent, x.querySelector('svg').outerHTML])""")
        nav = [(lab.lower(), lab, ico) for lab, ico in nav]
        gear = pg.locator(".lgear svg").evaluate("e => e.outerHTML"); mark = pg.locator(".lmark svg").evaluate("e => e.outerHTML")
        pg.close()
        sheets = []

        def screens(kind, dev, W, H):
            # One device's Play, game and Ask screens in frame `kind` (A rail, B top bar, phone: the app as it is).
            tag = f"{kind}-{dev}"
            if kind == "phone":
                for scr, nav_fn in (("1-play", setup), ("2-game", game), ("3-ask", ask_game)):
                    pg = page(b, W, H); nav_fn(pg); pg.wait_for_timeout(700); pg.screenshot(path=str(out / f"{tag}-{scr}.png")); pg.close()
                return
            mw, mh = (W - RAIL, H) if kind == "A" else (W, H - TOP)
            c = wclass(W)
            if c == "medium":
                plan = {"1-play": [(setup, mw)], "2-game": [(game, mw)], "3-ask": [(ask_game, mw)]}
            else:
                plan = {"1-play": [(setup, 460), (games_list, mw - 460)],
                        "2-game": ([(games_list, 340)] if c == "large" else []) + [(game, 460), (watch, mw - 460 - (340 if c == "large" else 0))],
                        "3-ask": [(game, 520 if c == "large" else 440), (ask_game, mw - (520 if c == "large" else 440))]}
            for scr, panes_plan in plan.items():
                panes = []
                for k, (fn, w) in enumerate(panes_plan):
                    z = max(1.0, min(w / 820, mh / 900, 1.4)) if fn is watch else 1.0
                    pg = page(b, w, mh, z); fn(pg); panes.append((shot(pg, f"{tag}-{scr}-{k}.png"), w, mh))
                sheets.append((f"{tag}-{scr}", kind, W, H, "ask" if scr == "3-ask" else "play", panes))

        for dev, _, W, H in DEVICES:
            screens("phone" if wclass(W) == "compact" else "A", dev, W, H)
        screens("B", "16in", 1728, 1035)
        # The duel last: the server keeps it, so every later page would land in it. Full screen on every device.
        for dev, _, W, H in DEVICES:
            z = play_zoom(W, H); d = page(b, W, H, z); duel(d)
            if dev == "phone":
                d.wait_for_timeout(500); d.screenshot(path=str(out / f"phone-{dev}-4-duel.png")); d.close()
            else:
                sheets.append((f"A-{dev}-4-duel", "none", W, H, "play", [(shot(d, f"A-{dev}-4-duel-0.png"), W, H)]))
        for name, kind, W, H, sel, panes in sheets:
            html = frame_html(kind, W, H, nav, sel, gear, mark, panes)
            m = b.new_page(viewport={"width": W, "height": H})
            m.set_content(html.replace("<head>", f'<head><base href="{BASE}/">', 1), wait_until="domcontentloaded"); m.wait_for_timeout(1500)
            m.screenshot(path=str(out / f"{name}.png")); m.close()
            print("mock", name, flush=True)
        b.close()
finally:
    srv.kill()
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
for scr, title in (("1-play", "Play"), ("2-game", "A saved game"), ("3-ask", "Ask about a game"), ("4-duel", "A duel")):
    ims = []
    for dev, lab, W, H in DEVICES:
        f = out / (f"phone-{dev}-{scr}.png" if dev == "phone" else f"A-{dev}-{scr}.png")
        ims.append((f"{lab} · {W}x{H}", Image.open(f)))
    th = 760; pad = 36
    rs = [im.resize((round(im.width * th / im.height), th), Image.LANCZOS) for _, im in ims]
    sheet = Image.new("RGB", (pad + sum(r.width + pad for r in rs), th + 130), "#0b0a09"); d = ImageDraw.Draw(sheet)
    d.text((pad, 22), "decklight · " + title + " on every screen", font=ImageFont.truetype(F, 30), fill="#d6cfc0")
    x = pad
    for (lab, _), r in zip(ims, rs):
        d.text((x, 80), lab, font=ImageFont.truetype(F, 19), fill="#958f82"); sheet.paste(r, (x, 110)); x += r.width + pad
    sheet.save(out / f"devices-{scr}.jpg", quality=85); print("strip", scr, flush=True)
