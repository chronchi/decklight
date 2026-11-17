"""The chat's game blocks in the real app against the real Rust server (web/features/ask/blocks.js).

Asks one question that makes the model call replay (summary and one turn) and bot_view on a saved game, then checks
the game card, the play sequence (play, step, the board changes) and the bot's view (rows, your pick, Watch the
turn), and screenshots each block. Uses this machine's saved OpenCode Go key (one answer). The games come from
`writes_demo_games` (rust/decklight-server/src/ask/replay.rs):

    ASK_DEMO_DIR=<dir> cargo test --release -p decklight-server writes_demo_games -- --ignored
    uv run --no-project --with playwright==1.57.0 python -u scripts/app/ask_blocks_check.py <dir>/records <out_dir>
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
records = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
if not (records / "index.sqlite").is_file():
    sys.exit(f"no saved games in {records}: run writes_demo_games first (see this script's docstring)")
ask_dir = ROOT / "data/cache/ask_blocks/ask"
shutil.rmtree(ask_dir, ignore_errors=True)
# bot_view runs the bot's model: the laptop's ONNX Runtime is the venv's.
env = dict(os.environ)
if "ORT_DYLIB_PATH" not in env:
    libs = sorted(ROOT.glob(".venv/lib/python3*/site-packages/onnxruntime/capi/libonnxruntime.so.*"))
    if not libs:
        sys.exit("no ONNX Runtime for bot_view: set ORT_DYLIB_PATH or `uv sync` (onnxruntime)")
    env["ORT_DYLIB_PATH"] = str(libs[-1])
log = (out / "server.log").open("wb")
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8813", "--token", "smoke",
                        "--ask-dir", str(ask_dir), "--records", str(records)], cwd=ROOT, env=env, stdout=log, stderr=log)
errs: list[str] = []
problems: list[str] = []
Q = ("Look at my last saved game. Call replay for the whole game, then replay turn 2, then bot_view at decision 2. "
     "Then answer in two short lines.")


def shot(el, name: str) -> None:
    el.scroll_into_view_if_needed()
    el.screenshot(path=str(out / f"{name}.png"))
    print("shot", name, flush=True)


try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
        pg.on("response", lambda r: errs.append(f"{r.status} {r.url}") if r.status >= 400 and not r.url.endswith("/__ready") else None)
        pg.goto("http://127.0.0.1:8813/?t=smoke&ltab=ask")
        pg.wait_for_selector(".bbar svg", timeout=60000)
        pg.click('.bbar button[data-tab="ask"]')
        pg.wait_for_selector(".ak-empty")
        pg.fill("#akin", Q)
        pg.click(".ak-send")
        pg.wait_for_selector(".ak-think", timeout=10000)
        t0 = time.time()
        while time.time() - t0 < 300 and not (pg.locator(".ak-think").count() == 0 and pg.locator(".ak-send:not(.stop)").count() == 1):
            pg.wait_for_timeout(500)
        pg.wait_for_timeout(1500)
        names = pg.eval_on_selector_all(".ak-trail li > button", "els => els.map(e => e.textContent)")
        print("calls:", names, flush=True)
        if pg.locator(".ak-err").count():
            problems.append("the answer failed: " + pg.inner_text(".ak-err"))
        blocks = pg.locator(".ab-rb")
        print("blocks:", blocks.count(), flush=True)
        if not blocks.count():
            problems.append("no game block under the answer")
        game = pg.locator('.ab-rb:has(.ab-spark)')
        if game.count():
            shot(game.first, "block-game")
        else:
            problems.append("no game card (did the model call replay for the whole game?)")
        # The sequence: wait for its boards, step it, check the caption and board move.
        seq = pg.locator('.ab-rb:has(.ab-scrub)').first
        if pg.locator('.ab-rb:has(.ab-scrub)').count():
            seq.scroll_into_view_if_needed()
            shot(seq, "block-seq-a")
            cap1 = seq.locator(".ab-cap").inner_text()
            n = seq.locator(".ab-n").inner_text()
            seq.locator('[data-do="next"]').click()
            pg.wait_for_timeout(400)
            seq = pg.locator('.ab-rb:has(.ab-scrub)').first
            cap2 = seq.locator(".ab-cap").inner_text()
            print("sequence", n, "|", cap1, "->", cap2, flush=True)
            if cap1 == cap2:
                problems.append("Next did not change the play's caption")
            seq.locator('[data-do="play"]').click()
            pg.wait_for_timeout(3500)
            seq = pg.locator('.ab-rb:has(.ab-scrub)').first
            shot(seq, "block-seq-b")
            print("after play:", seq.locator(".ab-n").inner_text(), flush=True)
        else:
            problems.append("no play sequence (did the model call replay with a turn?)" +
                            (" | " + pg.inner_text(".ab-fail") if pg.locator(".ab-fail").count() else ""))
        bot = pg.locator('.ab-rb:has(.ab-opts)')
        if bot.count():
            pg.wait_for_timeout(1000)
            shot(bot.first, "block-bot")
            rows = bot.first.locator(".ab-opt").count()
            if rows < 2:
                problems.append(f"the bot's view shows {rows} options")
            if not bot.first.locator(".ab-tag.ab-you").count():
                problems.append("the bot's view does not mark your pick")
            if not bot.first.locator(".ab-still").count():
                problems.append("the bot's view has no board")
            bot.first.locator('[data-act="ask:watch"]').click()
            pg.wait_for_timeout(2500)
            if not pg.locator('.ab-rb:has(.ab-opts) + .ab-rb .ab-scrub').count():
                problems.append("Watch the turn showed no sequence")
        else:
            problems.append("no bot's view block (did the model call bot_view?)")
        pg.screenshot(path=str(out / "thread.png"), full_page=True)
        b.close()
except Exception as e:  # noqa: BLE001
    problems.append(f"driver: {e}")
finally:
    srv.kill()

print("page errors:", errs)
print("problems:", problems)
sys.exit(1 if errs or problems else 0)
