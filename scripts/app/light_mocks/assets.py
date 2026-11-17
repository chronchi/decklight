"""Light theme (C Washi) with three card-back + logo treatments (2026-11-15; the user picked 1 Ai-zome).
Run from ygo-zero/: uv run --no-project --with playwright==1.57.0 python -u scripts/app/light_mocks/assets.py <out_dir>"""
import subprocess, sys, time, shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(Path(__file__).parent)); from pal import P
from shots import css, INJ
ROOT = Path.cwd(); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
base = css(P["C-washi"]).replace(".handcount i, .card.back, .thumb.back, .cue-card.back{--indigo-1:#22304a;--indigo-2:#161e2e}", "")
BACK = ".handcount i, .card.back, .thumb.back, .cue-card.back"
def wave(c):
    w = f"radial-gradient(circle at 50% 100%, transparent 0 2px, {c} 2px 3px, transparent 3px 5px, {c} 5px 6px, transparent 6px 8px, {c} 8px 9px, transparent 9px)"
    return f"background-image:{w},{w},linear-gradient(170deg, var(--indigo-1), var(--indigo-2)) !important;"
# The logo: three cards outlined in ink on paper, the path in a deeper kincha that reads on light.
def logo(fill, s1, s2, s3, gold):
    return (f".lmark rect{{fill:{fill}}} .lmark g:nth-of-type(1) rect{{stroke:{s1}}} .lmark g:nth-of-type(2) rect{{stroke:{s2}}}"
            f" .lmark g:nth-of-type(3) rect{{stroke:{s3}}} .lmark polyline{{stroke:{gold}}} .lmark circle[r='7.5']{{fill:{fill};stroke:{gold}}}"
            f" .lmark circle[r='13.8']{{fill:{gold}}}")
V = {
 "0-now": "",
 # 1 Ai-zome: indigo cloth washed pale; waves and border in a mid indigo, the diamond in kincha.
 "1-aizome": (f"{BACK}{{--indigo-1:#c3cede;--indigo-2:#a9b8cd;{wave('rgba(46,66,104,0.16)')}}}"
              ".card.back, .thumb.back, .cue-card.back{box-shadow:inset 0 0 0 2px #a9b8cd, inset 0 0 0 2.5px rgba(46,66,104,0.35), 0 0 0 1px rgba(42,39,33,0.18) !important}"
              ".card.back::after, .thumb.back::after, .cue-card.back::after{border-color:rgba(116,90,38,0.75)!important;background:rgba(116,90,38,0.14)!important}"
              ".pile .cnt{background:rgba(250,247,240,0.88);color:#2a2721}"
              + logo("#faf7f0", "#a9b8cd", "#7d8ea8", "#46628c", "#8a6a2c")),
 # 2 Paper print: the back is washi with indigo seigaiha printed on it and an indigo frame, like a hanafuda reverse.
 "2-print": (f"{BACK}{{--indigo-1:#f3efe5;--indigo-2:#ebe5d8;{wave('rgba(70,98,140,0.22)')}}}"
             ".card.back, .thumb.back, .cue-card.back{box-shadow:inset 0 0 0 3px #ebe5d8, inset 0 0 0 4px rgba(70,98,140,0.55), 0 0 0 1px rgba(42,39,33,0.22) !important}"
             ".card.back::after, .thumb.back::after, .cue-card.back::after{border-color:rgba(165,72,60,0.8)!important;background:rgba(165,72,60,0.12)!important}"
             ".pile .cnt{background:rgba(42,39,33,0.08);color:#2a2721}"
             + logo("#faf7f0", "#c4bcaa", "#8f887a", "#2a2721", "#8a6a2c")),
 # 3 Soft sumi: still a dark back (the card stays the physical card), lifted to a slate indigo so it sits on paper without a hole.
 "3-slate": (f"{BACK}{{--indigo-1:#56698a;--indigo-2:#43546f;{wave('rgba(225,232,245,0.10)')}}}"
             ".card.back, .thumb.back, .cue-card.back{box-shadow:inset 0 0 0 2px #43546f, inset 0 0 0 2.5px rgba(225,232,245,0.22), 0 0 0 1px rgba(42,39,33,0.2) !important}"
             ".card.back::after, .thumb.back::after, .cue-card.back::after{border-color:rgba(214,190,140,0.7)!important;background:rgba(214,190,140,0.14)!important}"
             ".pile .cnt{background:rgba(250,247,240,0.9);color:#2a2721}"
             + logo("#43546f", "#8f9bb0", "#b5bdcb", "#e9ebef", "#c9a865")),
}
for name, extra in V.items():
    if (out / f"{name}-duel.png").exists(): continue
    shutil.rmtree(out / "ask", ignore_errors=True); shutil.rmtree(out / "records", ignore_errors=True)
    srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8815", "--token", "smoke", "--ask-dir", str(out / "ask"),
                            "--records", str(out / "records")], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
    try:
        time.sleep(3)
        with sync_playwright() as pw:
            b = pw.chromium.launch(); pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
            pg.goto("http://127.0.0.1:8815/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
            pg.evaluate(INJ, base + extra); pg.wait_for_timeout(1200)
            pg.screenshot(path=str(out / f"{name}-lobby.png"), clip={"x": 0, "y": 0, "width": 412, "height": 130})
            pg.wait_for_selector('#startbtn:not([disabled])', timeout=60000); pg.click("#startbtn")
            for _ in range(12):
                if pg.locator('[data-act="gamemenu"]').count(): break
                for sel in ('[data-act="rpspick"]', '[data-act="rpsagain"]', '[data-act="rpsfirst"]', '[data-act="rpsbotfirst"]'):
                    if pg.locator(sel).count(): pg.locator(sel).first.click(); break
                pg.wait_for_timeout(1500)
            pg.wait_for_timeout(4000); pg.screenshot(path=str(out / f"{name}-duel.png"))
            b.close(); print("done", name, flush=True)
    finally:
        srv.kill(); srv.wait()
