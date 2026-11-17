"""Overlaps and overflow on the play screen and in Watch (user 2026-11-16: the turn label covered the left Extra
Monster Zone in a replay and the replay bar's speed button ran off the screen; "check for overflows in general").
Opens saved games in Watch at several phone widths, steps through it, then a live duel, and flags:
  - text of the turn panel (.phase) or the turn number (.turnno) over an Extra Monster Zone;
  - any element that sticks out of the screen sideways (outside a sideways scroller);
  - text cut off by its box (overflow hidden with no ellipsis) or spilling out of it;
  - in Watch on the user's phone (384 x 775 CSS px, 2026-11-16): your hand below the bottom of the player's frame, or a
    game menu button (a replay has no duel menu), or room between the board and Watch's header or controls; with its font scale (1.1), the turn panel or an Extra Monster Zone
    reaching out of the strip between the fields (it covered your field, 2026-11-16), and two very long player names
    pushing an LP bar off the screen;
  - in a duel's game menu on a short screen: rows that overlap (shrunk rows ran their text into the next), and a menu
    off centre at its first frame or while swiped down.
Games come from `writes_demo_games` (see games_check.py). Run from ygo-zero/ with ORT_DYLIB_PATH set:
    uv run --no-project --with playwright==1.57.0 python -u scripts/app/layout_check.py <dir>/records <out_dir>"""
import shutil, subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path.cwd(); records = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
if not (records / "index.sqlite").is_file():
    sys.exit(f"no saved games in {records}: run writes_demo_games first (see games_check.py)")
shutil.rmtree(out / "demo", ignore_errors=True); (out / "demo").mkdir()
shutil.copy(records.parent / "install_id", out / "demo" / "install_id")
records = Path(shutil.copytree(records, out / "demo" / "records", ignore=shutil.ignore_patterns("trash", "index.sqlite*")))
(records / "index.sqlite").touch()
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8832", "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(records)], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
WIDTHS = [320, 360, 384, 412]   # the narrowest phones, common Android, the user's S23 (font scale 1.1), the screenshots' width
PROBE = """() => {
  const vw = document.documentElement.clientWidth, found = [];
  const name = (el) => el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (typeof el.className === 'string' && el.className ? '.' + el.className.trim().split(/\\s+/).join('.') : '');
  const shown = (el) => { for (let e = el; e; e = e.parentElement) { const c = getComputedStyle(e); if (c.display === 'none' || c.visibility === 'hidden' || +c.opacity === 0) return false; } return true; };
  const scrollsX = (el) => { for (let e = el.parentElement; e; e = e.parentElement) { const o = getComputedStyle(e).overflowX; if ((o === 'auto' || o === 'scroll') && e.scrollWidth > e.clientWidth) return true; } return false; };
  const textOf = (el) => [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
  const inkBox = (el) => { const r = document.createRange(); r.selectNodeContents(el); return r.getBoundingClientRect(); };
  for (const el of document.body.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    if (r.width <= 2 || r.height <= 2 ||  // also the screen-reader-only text
         r.bottom < 0 || r.top > innerHeight || !shown(el)) continue;
    const c = getComputedStyle(el);
    if (c.position === 'fixed' && (r.left >= vw || r.right <= 0)) continue;  // a sheet parked off screen
    if ((r.right > vw + 1 || r.left < -1) && !scrollsX(el)) found.push('off screen: ' + name(el) + ' ' + Math.round(r.left) + '..' + Math.round(r.right) + ' of ' + vw);
    if (!textOf(el) || el.closest('.sr-only')) continue;
    const ink = inkBox(el);
    // Text that ends in "…" (text-overflow or a line clamp) or fades out on purpose (a covered hand card's name) is not cut.
    const marked = c.textOverflow === 'ellipsis' || (c.webkitLineClamp && c.webkitLineClamp !== 'none') ||
      (c.maskImage && c.maskImage !== 'none') || (c.webkitMaskImage && c.webkitMaskImage !== 'none');
    if (el.scrollWidth > el.clientWidth + 1 && (c.overflowX === 'hidden' || c.overflowX === 'clip') && !marked)
      found.push('text cut: ' + name(el) + ' in ' + name(el.closest('[id]') || el.parentElement) + ' "' + el.textContent.trim().slice(0, 30) + '"');
    else if (c.overflowX === 'visible' && (ink.right > r.right + 2 || ink.left < r.left - 2) && c.display !== 'inline')
      found.push('text spills: ' + name(el) + ' "' + el.textContent.trim().slice(0, 30) + '" ink ' + Math.round(ink.left) + '..' + Math.round(ink.right) + ' box ' + Math.round(r.left) + '..' + Math.round(r.right));
  }
  for (const t of document.querySelectorAll('#midstrip .phase > :not(.sr-only), #midstrip .turnno > *')) {
    const a = inkBox(t);
    for (const z of document.querySelectorAll('#midstrip .emz')) {
      const b = z.getBoundingClientRect();
      if (a.width && a.right > b.left + 1 && a.left < b.right - 1 && a.bottom > b.top && a.top < b.bottom)
        found.push('over the EMZ: ' + name(t) + ' "' + t.textContent.trim() + '" ink ' + Math.round(a.left) + '..' + Math.round(a.right) + ', zone ' + Math.round(b.left) + '..' + Math.round(b.right));
    }
  }
  // The strip between the fields holds the turn panel and the Extra Monster Zones: none may reach into a field.
  const m = document.getElementById('midstrip');
  if (m) {
    const mr = m.getBoundingClientRect();
    for (const t of m.querySelectorAll('.phase, .turnno, .emz')) {
      const a = t.classList.contains('emz') ? t.getBoundingClientRect() : inkBox(t);
      if (a.height && (a.bottom > mr.bottom + 1 || a.top < mr.top - 1))
        found.push('out of the middle strip: ' + name(t) + ' ' + Math.round(a.top) + '..' + Math.round(a.bottom) + ', strip ' + Math.round(mr.top) + '..' + Math.round(mr.bottom));
    }
  }
  return found;
}"""
# Android's font scale (the user's phone: 1.1) enlarges every px font size; Chromium has no switch for it, so the
# check scales the stylesheets' font sizes and lets the page fit again.
TEXT_ZOOM = """(k) => { for (const s of document.styleSheets) { let rs; try { rs = s.cssRules; } catch (e) { continue; }
  const walk = (rs) => { for (const r of rs) { if (r.cssRules) walk(r.cssRules); if (r.style && r.style.fontSize && r.style.fontSize.endsWith('px')) r.style.fontSize = (parseFloat(r.style.fontSize) * k) + 'px'; } };
  walk(rs); } dispatchEvent(new Event('resize')); }"""
problems, errs = {}, []


def probe(frame, where):
    for f in frame.evaluate(PROBE):
        problems.setdefault(f, set()).add(where)


def watch(pg, tag, w):
    # Watch: the first saved games, each over several turns (both players' names in the turn panel).
    pg.click('.lsub button[data-tab="games"]'); pg.wait_for_selector(".gm-row", timeout=30000)
    for g in range(min(3, pg.locator(".gm-row:not(:has(.gm-s))").count())):
        pg.locator(".gm-row:not(:has(.gm-s))").nth(g).click(); pg.wait_for_selector('[data-act="games:watch"]', timeout=30000)
        probe(pg.main_frame, f"{tag} game page")
        if g == 0: pg.screenshot(path=str(out / f"game-{w}.png"))
        pg.click('[data-act="games:watch"]'); pg.wait_for_selector("#watch .w-play", timeout=30000); pg.wait_for_timeout(800)
        if pg.locator('#watch [data-w="play"][aria-label="Pause"]').count(): pg.click('#watch [data-w="play"]')
        pg.frame_locator("#watch iframe").locator("#board").wait_for(timeout=30000)
        frame = pg.locator("#watch iframe").element_handle().content_frame()
        if w == 384:
            frame.evaluate(TEXT_ZOOM, 1.1); pg.wait_for_timeout(400)
            # Two very long names: they end in "…" and keep the LP bars' right side on the screen.
            frame.evaluate("""() => { const n = document.querySelectorAll('.lpblock .pname');
              if (n[0]) n[0].textContent = 'Dragonmaid Branded Despia (Labrynth variant)'; if (n[1]) n[1].textContent = 'carlos_the_duelist_from_brazil'; }""")
            probe(frame, f"{tag} watch board, long names")
        hb = frame.evaluate("() => { const h = document.querySelector('#hand'); return h ? [h.getBoundingClientRect().bottom, innerHeight] : null; }")
        if hb and hb[0] > hb[1] + 1:
            problems.setdefault(f"your hand runs {round(hb[0] - hb[1])} px below the Watch frame", set()).add(f"{tag} watch")
        # The board takes the whole screen; one bar over its bottom holds every control (Close far left, Export far right).
        gap = pg.evaluate("""() => { const b = document.querySelector('#watch iframe').getBoundingClientRect(), c = document.querySelector('#watch .w-bar').getBoundingClientRect();
          const r = [...document.querySelectorAll('#watch .w-row > button')].map(x => x.getAttribute('data-w'));
          return [b.top, innerHeight - b.bottom, c.bottom - innerHeight, r[0], r[r.length - 1], document.querySelector('#watch .w-top') ? 1 : 0]; }""")
        if max(gap[:3]) > 1 or gap[3:] != ["close", "export", 0]:
            problems.setdefault(f"Watch: board {round(gap[0])} px from the top, {round(gap[1])} px from the bottom, bar {round(gap[2])} px past it, row {gap[3]}..{gap[4]}, header {gap[5]}", set()).add(f"{tag} watch")
        # While it plays the bar hides, leaving the progress line; a tap on the board brings the bar back.
        pg.click('#watch [data-w="play"]'); pg.wait_for_timeout(600)
        st = pg.evaluate("() => { const w = document.getElementById('watch'); return [w.classList.contains('hidebar'), getComputedStyle(w.querySelector('.w-line')).opacity, getComputedStyle(w.querySelector('.w-bar')).pointerEvents]; }")
        pg.mouse.click(w // 2, 300); pg.wait_for_timeout(400)
        st2 = pg.evaluate("() => { const w = document.getElementById('watch'); return [w.classList.contains('hidebar'), w.querySelector('[data-w=play]').getAttribute('aria-label')]; }")
        if st != [True, "1", "none"] or st2 != [False, "Pause"]:
            problems.setdefault(f"Watch while playing: hidden {st}, after a tap {st2}", set()).add(f"{tag} watch")
        pg.click('#watch [data-w="play"]'); pg.wait_for_timeout(600)
        if pg.evaluate("() => document.getElementById('watch').classList.contains('hidebar')"):
            problems.setdefault("Watch keeps the bar hidden after a pause", set()).add(f"{tag} watch")
        if frame.locator('[data-act="gamemenu"]').count():
            problems.setdefault("a game menu button in Watch", set()).add(f"{tag} watch")
        for step in range(6):
            probe(pg.main_frame, f"{tag} watch"); probe(frame, f"{tag} watch board")
            if step == 1: pg.screenshot(path=str(out / f"watch-{w}-{g}.png"))
            nxt = pg.locator('#watch [data-w="last"]')
            if nxt.is_disabled(): break
            nxt.click(); pg.wait_for_timeout(500)
        pg.keyboard.press("Escape"); pg.wait_for_timeout(400)
        pg.click('.gm-head [aria-label="Back"]'); pg.wait_for_selector(".gm-row", timeout=30000)
    probe(pg.main_frame, f"{tag} games list")


def duel(pg):
    # A live duel (the play screen with "Your turn" / "Bot's turn") at every width. The server keeps the duel, so it
    # comes last: a reload would land back in it.
    pg.click('.lsub button[data-tab="play"]')
    pg.wait_for_selector('#startbtn:not([disabled])', timeout=60000); pg.click("#startbtn")
    for _ in range(12):
        if pg.locator('[data-act="gamemenu"]').count(): break
        for sel in ('[data-act="rpspick"]', '[data-act="rpsagain"]', '[data-act="rpsfirst"]', '[data-act="rpsbotfirst"]'):
            if pg.locator(sel).count(): pg.locator(sel).first.click(); break
        pg.wait_for_timeout(1500)
    pg.wait_for_selector('[data-act="gamemenu"]', timeout=60000); pg.wait_for_timeout(1500)
    for w in WIDTHS:
        pg.set_viewport_size({"width": w, "height": 800}); pg.wait_for_timeout(600)
        probe(pg.main_frame, f"{w}px duel"); pg.screenshot(path=str(out / f"duel-{w}.png"))
    # The game menu on a short screen: every row keeps its height (the body scrolls).
    pg.set_viewport_size({"width": 384, "height": 560}); pg.wait_for_timeout(600)
    pg.click('[data-act="gamemenu"]'); pg.wait_for_selector("#gamemenu .mbody", timeout=10000); pg.wait_for_timeout(500)
    rows = pg.evaluate("() => [...document.querySelectorAll('#gamemenu .mbody > *')].map(e => { const r = e.getBoundingClientRect(); return [r.top, r.bottom, e.scrollHeight, e.clientHeight, e.className]; })")
    for a, b2 in zip(rows, rows[1:]):
        if a[1] > b2[0] + 1:
            problems.setdefault(f"game menu rows overlap: {a[4]} / {b2[4]}", set()).add("384x560 duel")
    for r in rows:
        if r[2] > r[3] + 1:
            problems.setdefault(f"a game menu row is shorter than its text: {r[4]}", set()).add("384x560 duel")
    pg.screenshot(path=str(out / "gamemenu-short.png")); pg.keyboard.press("Escape")
    # The menu opens centred: frozen at its first frame, and dragged down by a swipe, it stays in the middle (its
    # keyframes once replaced translateX(-50%), so it came in from the bottom right; user 2026-11-16).
    pg.wait_for_timeout(400); pg.click('[data-act="gamemenu"]'); pg.wait_for_selector("#gamemenu .mbody", timeout=10000)
    x0 = pg.evaluate("() => { const m = document.getElementById('gamemenu'); for (const a of m.getAnimations()) { a.pause(); a.currentTime = 0; } const r = m.getBoundingClientRect(); return [r.left + r.width / 2, innerWidth / 2]; }")
    if abs(x0[0] - x0[1]) > 2:
        problems.setdefault(f"the game menu opens off centre ({x0[0]:.0f} px, centre {x0[1]:.0f})", set()).add("384x560 duel")
    hb = pg.locator("#gamemenu .mhead").bounding_box(); cx, cy = hb["x"] + 20, hb["y"] + hb["height"] / 2
    pg.mouse.move(cx, cy); pg.mouse.down(); pg.mouse.move(cx, cy + 40, steps=4)
    x1 = pg.evaluate("() => { const r = document.getElementById('gamemenu').getBoundingClientRect(); return r.left + r.width / 2; }")
    pg.mouse.move(cx, cy); pg.mouse.up()
    if abs(x1 - x0[1]) > 2:
        problems.setdefault(f"the game menu jumps sideways while swiped down ({x1:.0f} px, centre {x0[1]:.0f})", set()).add("384x560 duel")
    pg.keyboard.press("Escape")


try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": WIDTHS[0], "height": 800}, device_scale_factor=2)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto("http://127.0.0.1:8832/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
        try:
            for w in WIDTHS:
                pg.set_viewport_size({"width": w, "height": 775 if w == 384 else 800}); watch(pg, f"{w}px", w)
            duel(pg)
        except Exception:
            pg.screenshot(path=str(out / "failed.png")); raise
        b.close()
except Exception as e:  # noqa: BLE001
    problems.setdefault("driver: " + str(e).splitlines()[0] + " (screenshot failed.png)", set()).add("-")
finally:
    srv.kill()
for f, where in sorted(problems.items()):
    print(f, "|", ", ".join(sorted(where)))
print("page errors:", errs)
print(len(problems), "problems")
sys.exit(1 if problems or errs else 0)
