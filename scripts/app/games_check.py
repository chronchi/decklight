"""The Games screen (Play > Games, web/features/games/) in the real app against the real Rust server: the list with
its stats and the sample games, a game's page (game card, a turn's play sequence, its decisions), a decision's page
(the bot's view, all the options) and Check my decisions. No AI provider is used. The games come from
`writes_demo_games` (rust/decklight-server/src/ask/replay.rs):

    ASK_DEMO_DIR=<dir> cargo test --release -p decklight-server writes_demo_games -- --ignored
    uv run --no-project --with playwright==1.57.0 python -u scripts/app/games_check.py <dir>/records <out_dir>
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
# The check deletes a game: it runs on a copy (with the install id next to it, which says which seat is "you"), so
# the demo games stay as they are. The index holds absolute paths, so the copy's index is rebuilt from its files.
shutil.rmtree(out / "demo", ignore_errors=True)
(out / "demo").mkdir()
shutil.copy(records.parent / "install_id", out / "demo" / "install_id")
records = Path(shutil.copytree(records, out / "demo" / "records", ignore=shutil.ignore_patterns("trash", "index.sqlite*")))
(records / "index.sqlite").touch()
# The bot's view runs the bot's model: the laptop's ONNX Runtime is the venv's.
env = dict(os.environ)
if "ORT_DYLIB_PATH" not in env:
    libs = sorted(ROOT.glob(".venv/lib/python3*/site-packages/onnxruntime/capi/libonnxruntime.so.*"))
    if not libs:
        sys.exit("no ONNX Runtime for the bot's view: set ORT_DYLIB_PATH or `uv sync` (onnxruntime)")
    env["ORT_DYLIB_PATH"] = str(libs[-1])
log = (out / "server.log").open("wb")
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8814", "--token", "smoke",
                        "--ask-dir", str(out / "ask"), "--records", str(records)], cwd=ROOT, env=env, stdout=log, stderr=log)
errs: list[str] = []
problems: list[str] = []


def shot(pg, name: str) -> None:
    pg.screenshot(path=str(out / f"{name}.png"), full_page=True)
    print("shot", name, flush=True)


def need(pg, sel: str, what: str, timeout: int = 30000) -> bool:
    try:
        pg.wait_for_selector(sel, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001
        fail = pg.locator(".ab-fail").all_inner_texts()
        problems.append(f"{what}: {sel} did not appear" + (f" | {fail}" if fail else ""))
        return False


try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
        pg.on("response", lambda r: errs.append(f"{r.status} {r.url}") if r.status >= 400 and not r.url.endswith("/__ready") else None)
        pg.goto("http://127.0.0.1:8814/?t=smoke")
        pg.wait_for_selector(".bbar svg", timeout=60000)
        pg.click('.lsub button[data-tab="games"]')
        if need(pg, ".gm-row", "the list"):
            rows = pg.locator('.gm-row').count()
            samples = pg.locator('.gm-res.gm-s').count()
            print("rows", rows, "samples", samples, "stat:", pg.inner_text(".gm-big"), flush=True)
            if samples < 3:
                problems.append(f"{samples} sample games listed")
            if rows - samples < 3:
                problems.append(f"{rows - samples} of the 3 saved games listed")
            if not pg.locator(".gm-row svg").count():
                problems.append("no LP chart on the rows")
            shot(pg, "games-list")
            pg.click('.gm-chips [data-f="practice"]')
            need(pg, ".gm-stat", "the practice filter")
            if pg.locator('.gm-res.gm-w, .gm-res.gm-l').count():
                problems.append("the Practice filter lists duels")
            pg.click('.gm-chips [data-f="all"]')
            need(pg, ".gm-row", "the list again")
            pg.click('.gm-hide')
            if pg.locator('.gm-res.gm-s').count():
                problems.append("Hide sample games left them listed")
            pg.click('.gm-show')
        # A game: its card, a turn's sequence, the turn's decisions.
        pg.locator('.gm-row:not(:has(.gm-s))').first.click()
        if need(pg, ".gm-turn", "the game page"):
            if not pg.locator(".ab-spark").count():
                problems.append("no game card on the game page")
            turn = pg.locator('.gm-turn:has-text("You")').filter(has_text="decision").first
            turn.click()
            if need(pg, ".ab-scrub", "the turn's sequence"):
                cap1 = pg.inner_text(".ab-cap")
                pg.click('[data-act="games:seq"][data-do="next"]')
                pg.wait_for_timeout(300)
                if pg.inner_text(".ab-cap") == cap1:
                    problems.append("Next did not change the play")
            shot(pg, "games-game")
            if pg.locator(".gm-dec").count():
                pg.locator(".gm-dec").first.click()
                if need(pg, ".ab-opts", "the bot's view", 90000):
                    if not pg.locator(".gm-opt.you").count():
                        problems.append("the options list does not mark your pick")
                    # The board is the play screen's still (bf61c9f): it says "Drawing the board…" until the image is there.
                    try:
                        pg.wait_for_selector(".ab-still img", timeout=30000)
                    except Exception:  # noqa: BLE001
                        problems.append("the decision page's board was not drawn in 30 s")
                    shot(pg, "games-decision")
                    nxt = pg.locator('.gm-nav [data-act="games:decision"]').last
                    if nxt.count():
                        nxt.click()
                        need(pg, ".gm-ht h2", "the next decision")
                pg.click('[data-act="games:backgame"]')
            else:
                problems.append("no decisions under the open turn")
        # Check my decisions on the game (progress, then marks).
        if pg.locator('[data-act="games:check"]').count():
            pg.click('[data-act="games:check"]')
            t0 = time.time()
            saw = False
            while time.time() - t0 < 300 and not pg.locator('.gm-note[role="status"]').count():
                saw = saw or pg.locator('button:has-text("Checking")').count() > 0
                if pg.locator('.ab-fail').count():
                    problems.append("the check failed: " + pg.inner_text(".ab-fail"))
                    break
                pg.wait_for_timeout(500)
            print("check:", round(time.time() - t0, 1), "s; progress seen", saw, "|",
                  pg.locator('.gm-note[role="status"]').all_inner_texts(), flush=True)
            if not pg.locator('.gm-note[role="status"]').count():
                problems.append("Check my decisions did not finish")
            shot(pg, "games-checked")
        else:
            problems.append("no Check my decisions button")
        # Search and delete (mocks games-search, games-select, games-delete).
        pg.click('[data-act="games:back"]')
        if need(pg, "#gsearch", "the search box"):
            before = pg.locator('.gm-row:not(:has(.gm-s))').count()
            deck = pg.locator('.gm-row:not(:has(.gm-s)) .gm-ti').first.inner_text().split(" vs ")[0]
            pg.fill("#gsearch", deck[1:-1].upper())
            pg.wait_for_selector('.gm-note[role="status"]:has-text("match")', timeout=10000)
            print("search:", pg.inner_text('.gm-note[role="status"]'), flush=True)
            if pg.locator(".gm-stat").count():
                problems.append("the stat card stayed while searching")
            shot(pg, "games-search")
            pg.fill("#gsearch", "")
            pg.wait_for_selector(".gm-stat", timeout=10000)
            if not pg.locator('.gm-lblrow [data-act="games:select"]').count():
                problems.append("no Select button on the Your games row")
            # A long press on a row starts Select mode with that row ticked (and does not open the game).
            pg.locator('.gm-row[data-hold]').first.hover()
            pg.mouse.down()
            pg.wait_for_timeout(800)
            pg.mouse.up()
            if not pg.locator(".gm-selt").count() or "1 selected" not in pg.inner_text(".gm-selt"):
                problems.append("a long press did not start Select mode with the row ticked")
            if pg.locator(".gm-turn").count():
                problems.append("the long press also opened the game")
            pg.click('[data-act="games:selcancel"]')
            pg.click('[data-act="games:select"]')
            pg.locator('[data-act="games:tick"]').first.click()
            if "1 selected" not in pg.inner_text(".gm-selt"):
                problems.append("ticking a row did not select it")
            shot(pg, "games-select")
            pg.click('[data-act="games:delask"]')
            need(pg, '[role="alertdialog"]', "the delete dialog")
            pg.keyboard.press("Escape")
            if pg.locator('[role="alertdialog"]').count():
                problems.append("Escape did not close the delete dialog")
            pg.click('[data-act="games:delask"]')
            need(pg, '[role="alertdialog"]', "the delete dialog again")
            shot(pg, "games-delete")
            pg.click('[data-act="games:delyes"]')
            if need(pg, '.gm-note[role="status"]:has-text("Deleted 1 game")', "the deleted note"):
                pg.wait_for_selector(".gm-row", timeout=30000)
                after = pg.locator('.gm-row:not(:has(.gm-s))').count()
                print("rows before/after delete:", before, after, flush=True)
                if after != before - 1:
                    problems.append(f"the delete left {after} of {before} games listed")
                if not any((records / "trash").glob("*.match.json.gz")):
                    problems.append("no file in the records' trash folder")
        # Ask about this game: the Ask tab with the question in the composer, nothing sent.
        pg.locator('.gm-row:not(:has(.gm-s))').first.click()
        need(pg, '[data-act="games:askgame"]', "the game page again")
        pg.click('[data-act="games:askgame"]')
        need(pg, "#akin", "the Ask composer")
        if "Look at my game" not in pg.input_value("#akin"):
            problems.append("Ask about this game did not fill the composer")
        b.close()
except Exception as e:  # noqa: BLE001
    problems.append(f"driver: {e}")
finally:
    srv.kill()

print("page errors:", errs)
print("problems:", problems)
sys.exit(1 if errs or problems else 0)
