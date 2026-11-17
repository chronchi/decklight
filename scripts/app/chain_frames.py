"""Play a game in the web client against the real server and capture frame bursts while a chain is on screen (user
2026-11-13: "when there is a chain ... when a card goes to the field ... the other cards are overlaying it ... get a frame
by frame"). Our side picks Activate/Summon options first (to make chains), Pass otherwise.
Run from ygo-zero/: uv run --no-project --with playwright==1.57.0 python -u scripts/app/chain_frames.py <out> [you_deck] [bot_deck] [max_s]
Writes <out>/b<burst>-<frame>.png (one burst per chain, ~every 60 ms) and <out>/bursts.txt (start time, chain text)."""
import glob, os, pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
you = sys.argv[2] if len(sys.argv) > 2 else "mitsurugi-meta1"
bot = sys.argv[3] if len(sys.argv) > 3 else "sky-striker-meta1"
max_s = float(sys.argv[4]) if len(sys.argv) > 4 else 240
if not os.environ.get("ORT_DYLIB_PATH"):  # the product bot loads ONNX Runtime from the venv's onnxruntime wheel
    libs = sorted(glob.glob(".venv/lib/python3*/site-packages/onnxruntime/capi/libonnxruntime.so*"))
    if libs:
        os.environ["ORT_DYLIB_PATH"] = os.path.abspath(libs[-1])
srv = subprocess.Popen(["rust/target/release/decklight-serve", "--port", "8815", "--token", "frames"],
                       stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT)
PREF = ("Activate", "Special Summon", "Normal Summon", "Link Summon", "Xyz Summon", "Synchro Summon", "Fusion Summon")
errs, bursts = [], []
try:
    time.sleep(4)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=1.5, has_touch=True)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(f"http://127.0.0.1:8815/?t=frames&ltab=bot&bot={bot}"); pg.wait_for_timeout(3000)
        pg.locator("[data-pickopen='you']").first.click(); pg.wait_for_timeout(600)
        pg.click("[data-dsrc='prebuilt']"); pg.wait_for_timeout(300)
        pg.locator(f"[data-youdeck='{you}']").first.click(); pg.wait_for_timeout(600)
        if pg.locator("#decksheet [data-youdeck]").count():
            pg.locator("#decksheet .dkfoot [data-youdeck]").first.click(); pg.wait_for_timeout(400)
        pg.screenshot(path=str(out / "lobby.png"))
        pg.click("#startbtn"); pg.wait_for_timeout(3000)
        pg.screenshot(path=str(out / "start.png"))
        t0 = time.time(); last_burst_text = None; fails = 0
        while time.time() - t0 < max_s:
            if pg.locator("#chain").count():
                txt = pg.locator("#chain").inner_text().replace("\n", " ")
                k = len(bursts); bursts.append(f"{time.time()-t0:7.1f}s {txt}"); print("burst", k, txt, flush=True)
                n = 0
                while n < 60 and (pg.locator("#chain").count() or n < 25):
                    pg.screenshot(path=str(out / f"b{k:02d}-{n:02d}.png")); n += 1
                    if pg.locator("#sheet [data-opt]").count() and n > 3:
                        break
            if pg.locator(".result, #result, [data-act='rematch']").count():
                print("game over", flush=True); break
            rps = pg.locator("#rps button:visible")   # Rock / Throw again / Go first / Start duel
            if rps.count():
                rps.first.click(); pg.wait_for_timeout(800); continue
            zp = pg.locator("#board .zpick:visible")
            if zp.count():
                zp.first.click(); pg.wait_for_timeout(300); continue
            opts = pg.locator("#sheet [data-opt]:visible, #menu [data-opt]:visible")
            if opts.count():
                texts = opts.all_inner_texts()
                pick = next((i for i, t in enumerate(texts) if any(w in t for w in PREF) and "Pass" not in t), None)
                if pick is None:
                    pick = next((i for i, t in enumerate(texts) if t.strip().startswith("Pass") or "End the turn" in t), 0)
                try:
                    opts.nth(pick).click(timeout=2000)
                except Exception:
                    fails += 1
                pg.wait_for_timeout(250); continue
            lift = pg.locator("#hand .hcard.lift .card")
            if lift.count():
                try:
                    lift.first.click(timeout=2000); pg.wait_for_timeout(300)
                    m = pg.locator("#menu [data-opt]:visible")
                    if m.count():
                        texts = m.all_inner_texts()
                        pick = next((i for i, t in enumerate(texts) if any(w in t for w in PREF)), 0)
                        m.nth(pick).click(timeout=2000)
                except Exception:
                    fails += 1
                pg.wait_for_timeout(250); continue
            pg.wait_for_timeout(120)
        pg.screenshot(path=str(out / "end.png"))
        b.close()
finally:
    srv.terminate()
(out / "bursts.txt").write_text("\n".join(bursts) + "\n")
print("bursts", len(bursts), "errors", errs[:3])
sys.exit(1 if errs else 0)
