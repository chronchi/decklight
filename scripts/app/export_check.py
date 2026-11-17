"""Games > Export and Watch in the real app against the real Rust server: a game's Export sheet makes the EDOPro replay
and the text; a turn's still (the play screen) and its GIF; the still opens Watch, which steps a play on and exports an
image of one play and a video of a few turns, each saved as a browser download and checked (the .yrpX with
export_edopro_check.py's reader, the GIF with PIL, the MP4 with ffmpeg); Select mode exports two games' replays.
Escape closes the sheet, then Watch. No AI provider is used.

    ASK_DEMO_DIR=<dir> cargo test --release -p decklight-server writes_demo_games -- --ignored
    uv run --no-project --with playwright==1.57.0 --with pillow --with imageio-ffmpeg python -u scripts/app/export_check.py <dir>/records <out_dir>
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
records = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
if not (records / "index.sqlite").is_file():
    sys.exit(f"no saved games in {records}: run writes_demo_games first (see this script's docstring)")
shutil.rmtree(out / "demo", ignore_errors=True)
(out / "demo").mkdir()
shutil.copy(records.parent / "install_id", out / "demo" / "install_id")
records = Path(shutil.copytree(records, out / "demo" / "records", ignore=shutil.ignore_patterns("trash", "index.sqlite*")))
(records / "index.sqlite").touch()
env = dict(os.environ)
if "ORT_DYLIB_PATH" not in env:
    libs = sorted(ROOT.glob(".venv/lib/python3*/site-packages/onnxruntime/capi/libonnxruntime.so.*"))
    if libs:
        env["ORT_DYLIB_PATH"] = str(libs[-1])
log = (out / "server.log").open("wb")
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8817", "--token", "smoke",
                        "--ask-dir", str(out / "ask"), "--records", str(records)], cwd=ROOT, env=env, stdout=log, stderr=log)
errs: list[str] = []
problems: list[str] = []
files = out / "files"
shutil.rmtree(files, ignore_errors=True)
files.mkdir()


def shot(pg, name: str) -> None:
    pg.screenshot(path=str(out / f"{name}.png"))
    print("shot", name, flush=True)


def need(pg, sel: str, what: str, timeout: int = 60000) -> bool:
    try:
        pg.wait_for_selector(sel, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001
        fail = pg.locator(".gx .ab-fail, .ab-fail").all_inner_texts()
        problems.append(f"{what}: {sel} did not appear" + (f" | {fail}" if fail else ""))
        return False


def make(pg, what: str, fmt: str, name: str, timeout: int = 240000) -> list[Path]:
    """Pick the format, press Make, wait for Ready, press Share (a download in a browser); the saved files."""
    pg.click(f'.gx [data-act="games:xpas"][data-f="{fmt}"]')
    t0 = time.time()
    pg.click('.gx [data-act="games:xpmake"]')
    if not need(pg, '.gx [data-act="games:xpshare"]', f"{what} ready", timeout):
        shot(pg, f"fail-{name}")
        return []
    print(f"{what}: made in {time.time() - t0:.1f} s:", pg.inner_text(".gx .gx-ready").replace("\n", " | "), flush=True)
    shot(pg, f"ready-{name}")
    got = []
    n = pg.locator(".gx .gx-rn").inner_text()
    count = int(n.split()[0]) if n.endswith(" files") else 1
    with pg.expect_download() as d0:
        pg.click('.gx [data-act="games:xpshare"]')
    dls = [d0.value]
    for _ in range(count - 1):
        dls.append(pg.wait_for_event("download"))
    for d in dls:
        p = files / d.suggested_filename
        d.save_as(p)
        got.append(p)
    pg.click('.gx [data-act="games:xpback"]')
    return got


def ffprobe(p: Path) -> str:
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(p), "-f", "null", "-"], capture_output=True, text=True)
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(p)], capture_output=True, text=True).stderr
    if r.returncode or r.stderr.strip():
        problems.append(f"{p.name}: ffmpeg decode errors: {r.stderr.strip()[:300]}")
    return " ".join(line.strip() for line in info.splitlines() if "Duration" in line or "Video:" in line)


try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2, accept_downloads=True)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
        pg.goto("http://127.0.0.1:8817/?t=smoke")
        pg.wait_for_selector(".bbar svg", timeout=60000)
        pg.click('.lsub button[data-tab="games"]')
        need(pg, ".gm-row", "the list")
        pg.locator('.gm-row:not(:has(.gm-s))').first.click()
        need(pg, '[data-act="games:xpgame"]', "the game page's Export button")

        # The whole game as an EDOPro replay and as text.
        pg.click('[data-act="games:xpgame"]')
        need(pg, ".gx", "the Export sheet")
        shot(pg, "sheet")
        for f in make(pg, "EDOPro replay", "yrp", "yrp"):
            r = subprocess.run(["uv", "run", "python", "-c",
                                "import sys; sys.path.insert(0,'src'); from decklight.play.yrp import parse_yrpx; x=parse_yrpx(open(sys.argv[1],'rb').read()); "
                                "print(len(x.yrp.responses), x.yrp.names)", str(f)], cwd=ROOT, capture_output=True, text=True)
            print("yrpX:", f.name, f.stat().st_size, "B", r.stdout.strip(), r.stderr.strip()[-200:], flush=True)
            if r.returncode or not f.name.endswith(".yrpX"):
                problems.append(f"{f.name} does not parse as a .yrpX")
        for f in make(pg, "Text", "txt", "txt"):
            t = f.read_text()
            print("txt:", f.name, len(t.splitlines()), "lines; head:", t.splitlines()[0], flush=True)
            if "Turn 1 " not in t or "End:" not in t:
                problems.append(f"{f.name} lacks the turns or the end")
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(300)
        if pg.locator(".gx").count():
            problems.append("Escape did not close the Export sheet")

        # A turn: open the first turn with plays; its still is the play screen. A GIF of that turn from its export icon.
        turn = pg.locator(".gm-turn").nth(1)
        turn.click()
        need(pg, '[data-act="games:xpturn"]', "the turn's export icon")
        pg.click('[data-act="games:seq"][data-do="play"]')  # pause
        pg.click('[data-act="games:seq"][data-do="next"]')
        # The turn being stepped is the live board: the play screen itself in an iframe (not a still).
        if need(pg, ".ab-live .ab-lh iframe", "the turn's live board"):
            try:
                pg.frame_locator(".ab-lh iframe").locator("#board").wait_for(timeout=30000)
                shot(pg, "still")
            except Exception:  # noqa: BLE001
                problems.append("the turn's live board did not draw the play screen in 30 s")
        pg.click('[data-act="games:xpturn"]')
        need(pg, ".gx", "the Export sheet from the turn")
        for f in make(pg, "GIF of a turn", "gif", "gif"):
            im = Image.open(f)
            n = getattr(im, "n_frames", 1)
            print("gif:", f.name, f.stat().st_size, "B", im.size, n, "frames", flush=True)
            if im.size != (360, 640) or n < 2:
                problems.append(f"{f.name}: {im.size}, {n} frames")
            im.seek(n - 1)
            im.convert("RGB").save(out / "gif-last.png")
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(300)

        # Watch: the still opens the replay in the play screen at that play; step on, then export from it.
        pg.click(".ab-still")
        if need(pg, "#watch .w-play", "Watch's controls"):
            pg.wait_for_timeout(800)
            pos0 = pg.inner_text("#watch .w-pos")
            shot(pg, "watch")
            pg.click('#watch [data-w="next"]')
            pg.wait_for_timeout(2500)
            pg.click('#watch [data-w="next"]')
            pg.wait_for_timeout(2500)
            pos1 = pg.inner_text("#watch .w-pos")
            print("watch:", pos0, "->", pos1, flush=True)
            shot(pg, "watch-next")
            if pos0 == pos1:
                problems.append(f"Next play did not move Watch ({pos0})")
            board = pg.frame_locator("#watch iframe").locator("#board")
            if not board.count():
                problems.append("Watch shows no play-screen board")
            # The speed changed while a play is animating, then a step on: the board still moves (2026-11-16, the
            # user's phone: after 1x -> 2x the replay stopped and neither play nor the arrows changed the board).
            fr = pg.locator("#watch iframe").element_handle().content_frame()
            look = "() => document.getElementById('board').innerHTML + '|' + document.getElementById('info').innerText"
            # While it plays the bar is hidden: a tap on the board brings it back (as on the phone).
            reveal = lambda: pg.locator("#watch.hidebar").count() and pg.click("#watch .w-tap")
            pg.click('#watch [data-w="play"]'); pg.wait_for_timeout(700)
            reveal(); pg.click('#watch [data-w="speed"]'); pg.wait_for_timeout(2500)
            if pg.locator('#watch [data-w="play"][aria-label="Pause"]').count():
                reveal(); pg.click('#watch [data-w="play"]')
            pg.wait_for_timeout(1500)
            pos2, b2 = pg.inner_text("#watch .w-pos"), fr.evaluate(look)
            pg.click('#watch [data-w="next"]'); pg.wait_for_timeout(2500)
            pos3, b3 = pg.inner_text("#watch .w-pos"), fr.evaluate(look)
            print("watch after 2x:", pos1, "->", pos2, "->", pos3, flush=True)
            if pos1 == pos2 or pos2 == pos3 or b2 == b3:
                problems.append(f"after the speed change Watch stopped moving ({pos1} -> {pos2} -> {pos3}, board changed: {b2 != b3})")
            pg.click('#watch [data-w="speed"]'); pg.wait_for_timeout(300)
            pg.click('#watch [data-w="export"]')
            need(pg, "#watch .gx", "the Export sheet over Watch")
            pg.click('.gx [data-act="games:xpwhat"][data-v="moment"]')
            for f in make(pg, "Image of one play", "png", "png"):
                im = Image.open(f)
                print("png:", f.name, im.size, flush=True)
                if im.size != (1080, 1920):
                    problems.append(f"{f.name}: {im.size}")
                shutil.copy(f, out / "frame.png")
            # A few turns as video.
            pg.click('.gx [data-act="games:xpwhat"][data-v="range"]')
            shot(pg, "range")
            for f in make(pg, "Video of turns", "mp4", "mp4", timeout=600000):
                print("mp4:", f.name, f.stat().st_size, "B", ffprobe(f), flush=True)
            pg.keyboard.press("Escape")
            pg.wait_for_timeout(300)
            if pg.locator("#watch .gx").count():
                problems.append("Escape did not close the Export sheet over Watch")
            if not pg.locator("#watch").count():
                problems.append("Escape closed Watch together with the sheet")
            pg.keyboard.press("Escape")
            pg.wait_for_timeout(300)
            if pg.locator("#watch").count():
                problems.append("Escape did not close Watch")

        # Select mode: two games' replays at once.
        pg.click('[data-act="games:back"]')
        need(pg, '[data-act="games:select"]', "Select")
        pg.click('[data-act="games:select"]')
        ticks = pg.locator('[data-act="games:tick"]')
        ticks.nth(0).click()
        ticks.nth(1).click()
        pg.click('[data-act="games:xpsel"]')
        need(pg, ".gx", "the Export sheet from Select")
        shot(pg, "select-sheet")
        got = make(pg, "Two EDOPro replays", "yrp", "yrp2")
        print("select export:", [f.name for f in got], flush=True)
        if len(got) != 2:
            problems.append(f"Select export gave {len(got)} files, not 2")
        # Card art that never arrives (a stalled connection, 2026-11-16: the chat's boards stayed on "Drawing the
        # board…" for good): on a page whose fetches of card art hang (as a stalled request does: only an abort ends
        # them), a turn's live board is drawn anyway, without art, in time.
        pg.close()   # the server serves one app at a time: a second page takes the connection over
        hang = b.new_page(viewport={"width": 412, "height": 915})
        hang.add_init_script("""(() => { const f = window.fetch;
          window.fetch = (u, o) => String(u && u.url || u).includes('/img/') ? new Promise((_, rej) => {
            if (o && o.signal) o.signal.addEventListener('abort', () => rej(new DOMException('aborted', 'AbortError'))); }) : f(u, o); })()""")
        hang.goto("http://127.0.0.1:8817/?t=smoke"); hang.wait_for_selector(".bbar svg", timeout=60000)
        hang.click('.lsub button[data-tab="games"]'); hang.wait_for_selector(".gm-row", timeout=30000)
        hang.locator('.gm-row:not(:has(.gm-s))').first.click(); hang.wait_for_selector(".gm-turn", timeout=30000)
        hang.locator(".gm-turn").nth(1).click()
        t0 = time.time()
        try:
            hang.wait_for_selector(".ab-lh iframe, .ab-fail", timeout=45000)
            if hang.locator(".ab-fail").count():
                problems.append("a turn's live board failed when card art hangs (it should draw without art): " + hang.inner_text(".ab-fail"))
            else:
                hang.frame_locator(".ab-lh iframe").locator("#board").wait_for(timeout=45000)
                print(f"live board with hanging art: drawn in {time.time() - t0:.1f} s", flush=True)
        except Exception:  # noqa: BLE001
            problems.append("a turn's live board was not drawn in 45 s when card art hangs")
        hang.close()
        b.close()
except Exception as e:  # noqa: BLE001
    problems.append(f"driver: {e}")
finally:
    srv.kill()

print("page errors:", errs)
print("problems:", problems)
sys.exit(1 if errs or problems else 0)
