"""How fast a game page's turn block steps while it plays (user 2026-11-16, phone: "As it transitions from one frame to
the next it takes some seconds, not cool ... It should be near instant"). Opens a game, opens a turn of yours (it
plays by itself) and logs the time between steps; the block steps every 1.6 s when nothing holds it. Chromium's CPU
is slowed `--cpu` times (4 by default) to stand in for a phone. Prints the gaps; exits 1 when the median is > 2.0 s.
Games come from `writes_demo_games` (see games_check.py). Run from ygo-zero/:
    uv run --no-project --with playwright==1.57.0 python -u scripts/app/seq_speed.py <dir>/records <out_dir> [--cpu 4]"""
import shutil, statistics, subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path.cwd(); records = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
cpu = float(sys.argv[sys.argv.index("--cpu") + 1]) if "--cpu" in sys.argv else 4.0
shutil.rmtree(out / "demo", ignore_errors=True); (out / "demo").mkdir()
shutil.copy(records.parent / "install_id", out / "demo" / "install_id")
records = Path(shutil.copytree(records, out / "demo" / "records", ignore=shutil.ignore_patterns("trash", "index.sqlite*")))
(records / "index.sqlite").touch()
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8853", "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(records)], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
code = 1
try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 384, "height": 775})
        errs = []; pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("console", lambda m: m.type == "error" and "404" not in m.text and errs.append(m.text))
        pg.on("response", lambda r: r.status >= 400 and not r.url.endswith("/__ready") and errs.append(f"{r.status} {r.url}"))  # /__ready: the desktop shell's, 404 here by design
        cdp = pg.context.new_cdp_session(pg); cdp.send("Emulation.setCPUThrottlingRate", {"rate": cpu})
        pg.goto("http://127.0.0.1:8853/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=120000)
        pg.click('.lsub button[data-tab="games"]'); pg.wait_for_selector(".gm-row", timeout=60000)
        pg.locator(".gm-row:not(:has(.gm-s))").first.click(); pg.wait_for_selector(".gm-turn", timeout=60000)
        pg.locator('.gm-turn:has-text("You")').filter(has_text="decision").first.click()
        pg.wait_for_selector(".ab-rb .ab-n", timeout=60000)
        t0 = time.time(); last = None; marks = []
        while time.time() - t0 < 90:
            n = pg.evaluate("() => { const e = document.querySelector('.ab-rb .ab-n'); return e ? e.textContent : null; }")
            if n != last:
                marks.append((time.time() - t0, n)); last = n
                print(f"{marks[-1][0]:6.2f} s  {n}", flush=True)
                a, z = n.split(" / ")
                if a == z: break
            pg.wait_for_timeout(50)
        pg.screenshot(path=str(out / "seq.png")); print("page errors:", errs, flush=True)
        gaps = [round(marks[k][0] - marks[k - 1][0], 2) for k in range(1, len(marks))]
        print(f"cpu x{cpu}: steps {len(gaps)}, gaps {gaps}", flush=True)
        if gaps:
            med = statistics.median(gaps); print(f"median {med:.2f} s, max {max(gaps):.2f} s", flush=True)
            code = 0 if med <= 2.0 and not errs else 1
        else:
            print("ALERT: the turn never stepped", flush=True)
        b.close()
finally:
    srv.kill()
sys.exit(code)
