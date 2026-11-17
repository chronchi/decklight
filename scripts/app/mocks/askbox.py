"""Mock for Ask's question box (user 2026-11-16, phone screenshot of a 3-line question cut to one line: "when I'm typing
in the ask about cards the box outline keeps flickering ... when the text spans multiple lines it gets cut, it should
be somehow similar to the app that I attached here, that has the attachment then it shows up to 6 lines ... and then
has the send tool at the bottom"). On the user's phone size (384 x 775, font scale 1.1):
  - now: + | a pill-shaped box | send, in one row;
  - proposed: one rounded card holding the attachments, then the question (1 to 6 lines, then it scrolls), then a row
    with + at the left and send at the right. Empty, three lines with a card attached, and nine lines.
The user then sent the Claude app's empty box ("This is how it should look like empty"): one row, + | the question |
send, while the question fits on a line; then, of the Claude app's box with its model pill: "I think we can then add
the pill for the model selection like in the attached figure". Built: always the card, its bottom row + | the model
in use (opens the model sheet; it left the header) | send. This script now shoots the app's own box (the first
version injected the mock's CSS).
Run from ygo-zero/:
    uv run --no-project --with playwright==1.57.0 --with pillow python -u scripts/app/mocks/askbox.py <out_dir>"""
import json, subprocess, sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path.cwd(); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / "ask").mkdir(exist_ok=True); (out / "records").mkdir(exist_ok=True)
(out / "ask/settings.json").write_text(json.dumps({"preset": "custom", "model": "fake", "base_url": "http://127.0.0.1:9/v1"}))
ZOOM = """(k) => { for (const s of document.styleSheets) { let rs; try { rs = s.cssRules; } catch (e) { continue; }
  const walk = (rs) => { for (const r of rs) { if (r.cssRules) walk(r.cssRules); if (r.style && r.style.fontSize && r.style.fontSize.endsWith('px')) r.style.fontSize = (parseFloat(r.style.fontSize) * k) + 'px'; } };
  walk(rs); } }"""
THREE = "Esse Ghost Ogre pode destruir um Spell contínuo que já estava virado para cima?\nE se ele for negado?"
NINE = "\n".join(["Linha %d da pergunta, só para ver o limite" % i for i in range(1, 10)])
# The app's own box, no injected CSS. Each state checks the box after a render (attaching a card
# re-renders the tab): the question must keep its height and lines.
OPTS = [("empty", "empty", "", False), ("short", "one line", "What does Maxx C do?", False), ("three", "3 lines", THREE, False),
        ("three-card", "3 lines, then a card", THREE, True), ("nine", "9 lines (6 shown)", NINE, False)]


def label(img_path, text):
    im = Image.open(img_path).convert("RGB")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    sheet = Image.new("RGB", (im.width, im.height + 64), "#0f0e0c")
    sheet.paste(im, (0, 64)); ImageDraw.Draw(sheet).text((20, 16), text, font=font, fill="#d6cfc0")
    return sheet


srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8851", "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(out / "records")], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        problems = []
        for opt, _, text, card in OPTS:
            pg = b.new_page(viewport={"width": 384, "height": 775}, device_scale_factor=2, has_touch=True)
            pg.goto("http://127.0.0.1:8851/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
            pg.click('[data-tab="ask"]'); pg.wait_for_selector(".ak-plus"); pg.wait_for_timeout(800)
            pg.evaluate(ZOOM, 1.1)
            if text:
                pg.click("#akin"); pg.keyboard.insert_text(text); pg.wait_for_timeout(300)
            look = "() => { const t = document.getElementById('akin'); return [Math.round(t.getBoundingClientRect().height), !!document.querySelector('.ak-box .ak-model'), t.scrollHeight > t.clientHeight + 1]; }"
            before = pg.evaluate(look)
            if card:
                pg.tap(".ak-plus"); pg.tap('[data-act="ask:attach-card"]'); pg.fill("#akattq", "Ghost Ogre")
                pg.wait_for_selector('[data-act="ask:attach-pick"][data-kind="card"]', timeout=15000); pg.wait_for_timeout(300)
                pg.tap('[data-act="ask:attach-pick"][data-kind="card"]'); pg.wait_for_timeout(500)
                after = pg.evaluate(look)
                if after[0] != before[0]:
                    problems.append(f"{opt}: the question was {before} before the render, {after} after")
            st = pg.evaluate(look); print(opt, "height, model pill, scrolls:", st, flush=True)
            if not st[1] or st[2] != (opt == "nine"):
                problems.append(f"{opt}: model pill {st[1]}, scrolls {st[2]}")
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(out / f"{opt}.png")); pg.close()
        b.close()
        print("problems:", problems, flush=True)
    ims = [label(out / f"{o}.png", t) for o, t, _, _ in OPTS]
    sheet = Image.new("RGB", (sum(i.width for i in ims) + 20 * (len(ims) - 1), ims[0].height), "#0f0e0c")
    x = 0
    for i in ims:
        sheet.paste(i, (x, 0)); x += i.width + 20
    sheet.save(out / "askbox-built.jpg", quality=82); print("sheet", out / "askbox-built.jpg", flush=True)
finally:
    srv.kill()
