"""Ask tab in the real app against the real Rust server (docs/plans/ask-tab.md step 7).

Drives the web client in Chromium at phone size: bottom bar, Play sub-tabs, Ask empty state, one answered question
(stream, thinking line, tool trail, output), Retry (a second answer version with ‹ 1/2 ›), chat list, model sheet,
Settings. Uses this machine's saved OpenCode Go key (two short model calls). Chats go to a scratch folder.

    uv run --no-project --with playwright==1.57.0 python -u scripts/app/ask_ui_check.py <out_dir>
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
ask_dir = ROOT / "data/cache/ask_ui/ask"
shutil.rmtree(ask_dir, ignore_errors=True)
log = (out / "server.log").open("wb")
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8812", "--token", "smoke",
                        "--ask-dir", str(ask_dir), "--records", str(ROOT / "data/cache/ask_ui/records")],
                       cwd=ROOT, stdout=log, stderr=log)
errs: list[str] = []
problems: list[str] = []
n = 0


def shot(pg, name: str) -> None:
    global n
    n += 1
    pg.screenshot(path=str(out / f"{n:02d}-{name}.png"))
    print(f"shot {n} {name}", flush=True)


def wait_answer(pg, what: str, timeout_s: float = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if pg.locator(".ak-think").count() == 0 and pg.locator(".ak-send:not(.stop)").count() == 1:
            return
        pg.wait_for_timeout(500)
    problems.append(f"{what}: no answer in {timeout_s:.0f} s")


try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        # 4xx/5xx by URL, except /__ready (the app's start-up probe, only answered by the Tauri gate).
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
        pg.on("response", lambda r: errs.append(f"{r.status} {r.url}") if r.status >= 400 and not r.url.endswith("/__ready") else None)
        pg.goto("http://127.0.0.1:8812/?t=smoke&ltab=play")
        pg.wait_for_selector(".bbar svg", timeout=60000)
        pg.wait_for_timeout(800)
        shot(pg, "play-bot")
        pg.click('.lsub button[data-tab="people"]')
        pg.wait_for_timeout(400)
        shot(pg, "play-people")
        pg.click('.bbar button[data-tab="ask"]')
        pg.wait_for_selector(".ak-empty")
        pg.wait_for_timeout(600)
        shot(pg, "ask-empty")
        pg.fill("#akin", "How many Ash Blossom should my Branded deck run? Check the draw odds for 2 and 3 copies, answer in 3 short lines.")
        pg.click(".ak-send")
        pg.wait_for_selector(".ak-think", timeout=10000)
        pg.wait_for_timeout(1200)
        shot(pg, "ask-thinking")
        for _ in range(60):  # catch a tool on screen if there is one
            if pg.locator(".ak-trail .now").count():
                shot(pg, "ask-tool-running")
                break
            if pg.locator(".ak-think").count() == 0:
                break
            pg.wait_for_timeout(250)
        wait_answer(pg, "first question")
        pg.wait_for_timeout(500)
        shot(pg, "ask-answer")
        if not pg.locator(".ak-a").count() and not pg.locator(".ak-err").count():
            problems.append("no answer text and no error after the first question")
        box = pg.locator("#akin").bounding_box()
        if box and box["height"] > 50:
            problems.append(f"the composer is {box['height']:.0f} px tall when empty (two lines?)")
        linked = pg.eval_on_selector_all(".ak-a .ak-card:not(.unres)", "els => els.map(e => e.textContent)")
        unres = pg.eval_on_selector_all(".ak-a .ak-card.unres", "els => els.map(e => e.textContent)")
        print("linked cards:", linked, "| not a card name:", unres, flush=True)
        if "[[" in pg.inner_text(".ak-thread"):
            problems.append("an answer shows raw [[ ]] marks")
        if pg.locator(".ak-a .ak-card:not(.unres)").count():
            pg.click(".ak-a .ak-card:not(.unres) >> nth=0")
            pg.wait_for_timeout(1500)
            shot(pg, "ask-card-detail")
            if not pg.inner_text("#ov-detail").strip():
                problems.append("tapping a card name opened no details")
            pg.go_back()
            pg.wait_for_timeout(500)
        else:
            print("note: the answer named no card from its tool results", flush=True)
        # An unmatched [[name]] (option 1): a dotted link that asks which card it is.
        pg.evaluate("""document.querySelector('.ak-a').insertAdjacentHTML('beforeend',
            '<button class="ak-card unres" id="probe" data-act="ask:find" data-name="Veiler">Veiler</button>')""")
        pg.click("#probe")
        pg.wait_for_selector(".ak-bsheet .ak-mrow", timeout=10000)
        shot(pg, "ask-which-card")
        rows = pg.eval_on_selector_all(".ak-bsheet .ak-mrow .mn", "els => els.map(e => e.textContent)")
        if "Effect Veiler" not in rows:
            problems.append(f"which-card sheet for 'Veiler' lacks Effect Veiler: {rows}")
        else:
            pg.click('.ak-bsheet .ak-mrow:has-text("Effect Veiler")')
            pg.wait_for_timeout(1200)
            if "Effect Veiler" not in pg.inner_text("#ov-detail"):
                problems.append("picking Effect Veiler did not open its details")
            pg.go_back()
            pg.wait_for_timeout(500)
        pg.evaluate("document.querySelector('#lobby.askview .lscroll').scrollTop = 0")
        pg.wait_for_timeout(500)
        if pg.locator(".ak-jump:visible").count():
            shot(pg, "ask-scrolled-up")
            pg.click(".ak-jump")
            pg.wait_for_timeout(1200)
            if pg.locator(".ak-jump:visible").count():
                problems.append("the jump button stayed after jumping")
        elif pg.evaluate("(() => { const s = document.querySelector('#lobby.askview .lscroll'); return s.scrollHeight - s.clientHeight > 300; })()"):
            problems.append("scrolled up but no jump-to-latest button")
        if pg.locator(".ak-trail .sum").count():
            pg.click(".ak-trail .sum")
            pg.wait_for_timeout(300)
            pg.click(".ak-trail li > button >> nth=0")
            pg.wait_for_timeout(300)
            shot(pg, "ask-tool-output")
            if not pg.locator(".ak-trail .out").count():
                problems.append("tapping a call showed no output")
        pg.click('.ak-meta .act[aria-label="Retry"]')
        pg.wait_for_selector(".ak-think", timeout=10000)
        wait_answer(pg, "retry")
        pg.wait_for_timeout(500)
        shot(pg, "ask-retry")
        if "2 / 2" not in pg.inner_text(".ak-thread"):
            problems.append("Retry did not show the answer as version 2 / 2")
        pg.click('.ak-branch button[data-dir="-1"]')
        pg.wait_for_timeout(1200)
        if "1 / 2" not in pg.inner_text(".ak-thread"):
            problems.append("‹ did not switch to answer 1 / 2")
        shot(pg, "ask-branch-1")
        pg.click('button[aria-label="Chats"]')
        pg.wait_for_selector(".ak-chat", timeout=10000)
        shot(pg, "ask-list")
        pg.click(".ak-chat .open")
        pg.wait_for_timeout(600)
        pg.click(".ak-model")
        pg.wait_for_timeout(4000)
        shot(pg, "ask-model-sheet")
        if not pg.locator(".ak-mrow").count():
            problems.append("the model sheet listed no models")
        pg.click(".ak-scrim", position={"x": 200, "y": 40})
        pg.click('.ak-q >> nth=0')
        pg.wait_for_timeout(300)
        shot(pg, "ask-q-menu")
        pg.click(".ak-scrim", position={"x": 200, "y": 40})
        pg.click(".ak-model")
        pg.wait_for_timeout(300)
        pg.click('[data-act="ask:settings"]')
        pg.wait_for_selector(".ak-prov", timeout=10000)
        shot(pg, "ask-settings")
        if "key ••••" not in pg.inner_text(".ak-set"):
            problems.append("Settings did not show the saved key's hint")
        b.close()
except Exception as e:  # noqa: BLE001
    problems.append(f"driver: {e}")
finally:
    srv.kill()

print("page errors:", errs)
print("problems:", problems)
sys.exit(1 if errs or problems else 0)
