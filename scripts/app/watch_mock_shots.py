"""Mock shots of Watch (a saved game in the play screen, docs/plans/export.md): the real web client in replay mode
shows `games:watch` frames, with the proposed transport bar injected for the mock only. No AI provider is used.

    DECKLIGHT_WATCH_DUMP=<dir>/watch.json cargo test --release -p decklight-server watch_frames
    uv run --no-project --with playwright==1.57.0 python -u scripts/app/watch_mock_shots.py <dir>/watch.json <out_dir>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
w = json.loads(src.read_text())
frames = w["frames"]
if not frames:
    sys.exit(f"no frames in {src}")

ICON = {
    "first": '<path d="M6 5v10M15 5l-6 5 6 5"/>',
    "prev": '<path d="M12.5 5l-5 5 5 5"/>',
    "play": '<path d="M7 5l8 5-8 5z"/>',
    "next": '<path d="M7.5 5l5 5-5 5"/>',
    "last": '<path d="M14 5v10M5 5l6 5-6 5"/>',
    "share": '<path d="M10 3v10M6 7l4-4 4 4M4 12v4h12v-4"/>',
    "close": '<path d="M5 5l10 10M15 5 5 15"/>',
}
CSS = """
#wbar { padding: 0 12px 12px; }
#wbar .wprog { position: relative; height: 3px; margin: 0 6px 8px; border-radius: 2px; background: var(--line-2); }
#wbar .wprog > i { position: absolute; inset: 0 auto 0 0; border-radius: 2px; background: var(--accent); opacity: .75; }
#wbar .wprog > b { position: absolute; top: -3px; width: 1px; height: 9px; background: var(--faint); }
#wbar .wrow { display: flex; align-items: center; gap: 2px; }
#wbar button { min-width: 44px; min-height: 44px; border: 0; border-radius: 10px; background: none; color: var(--text);
  display: inline-flex; align-items: center; justify-content: center; }
#wbar button svg { width: 20px; height: 20px; fill: none; stroke: currentColor; stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }
#wbar button.wplay { border: 1px solid rgba(184,154,94,.6); background: var(--accent-tint); color: var(--accent); }
#wbar button.wplay svg { fill: currentColor; stroke: none; }
#wbar .wpos { flex: 1; text-align: center; font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }
#wbar .wsp { min-width: 44px; font-size: 13px; color: var(--muted); }
#wtop { position: fixed; top: 6px; right: 8px; z-index: 30; display: flex; gap: 4px; }
#wtop button { min-width: 44px; min-height: 44px; border: 1px solid var(--line-2); border-radius: 22px; background: rgba(20,19,16,.85);
  color: var(--text); display: inline-flex; align-items: center; justify-content: center; gap: 6px; padding: 0 12px; font-size: 14px; }
#wtop svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }
"""


def icon(name: str) -> str:
    return f'<svg viewBox="0 0 20 20" aria-hidden="true">{ICON[name]}</svg>'


def bar(i: int) -> str:
    f = frames[i]
    turn = f["turn"]
    in_turn = [k for k, x in enumerate(frames) if x["turn"] == turn]
    pos = in_turn.index(i) + 1
    last = w["last_turn"]
    pct = 100 * i / max(1, len(frames) - 1)
    ticks = "".join(f'<b style="left:{100 * k / max(1, len(frames) - 1):.2f}%"></b>'
                    for k in range(1, len(frames)) if frames[k]["turn"] != frames[k - 1]["turn"])
    return (f'<div id="wbar"><div class="wprog" aria-hidden="true"><i style="width:{pct:.1f}%"></i>{ticks}</div>'
            '<div class="wrow">'
            f'<button aria-label="Start of turn">{icon("first")}</button>'
            f'<button aria-label="Previous play">{icon("prev")}</button>'
            f'<button class="wplay" aria-label="Play">{icon("play")}</button>'
            f'<button aria-label="Next play">{icon("next")}</button>'
            f'<button aria-label="Next turn">{icon("last")}</button>'
            f'<span class="wpos">Turn {turn} of {last} · {pos}/{len(in_turn)}</span>'
            '<button class="wsp" aria-label="Speed: normal">1×</button></div></div>')


TOP = (f'<div id="wtop"><button aria-label="Export">{icon("share")}Export</button>'
       f'<button aria-label="Close the replay">{icon("close")}</button></div>')

env = dict(os.environ)
if "ORT_DYLIB_PATH" not in env:
    libs = sorted(ROOT.glob(".venv/lib/python3*/site-packages/onnxruntime/capi/libonnxruntime.so.*"))
    if libs:
        env["ORT_DYLIB_PATH"] = str(libs[-1])
log = (out / "server.log").open("wb")
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8818", "--token", "smoke",
                        "--ask-dir", str(out / "ask")], cwd=ROOT, env=env, stdout=log, stderr=log)
errs: list[str] = []
try:
    for _ in range(120):
        try:
            import urllib.request
            urllib.request.urlopen("http://127.0.0.1:8818/?t=smoke", timeout=1)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    body = json.dumps(frames)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route(lambda u: u.split("?")[0].endswith("/__watch.json"), lambda r: r.fulfill(status=200, content_type="application/json", body=body))
        pg.goto("http://127.0.0.1:8818/?t=smoke&replay=/__watch.json")
        pg.on("console", lambda m: print("console:", m.text, flush=True))
        try:
            pg.wait_for_function("window.frameCount > 0", timeout=30000)
        except Exception:  # noqa: BLE001
            pg.screenshot(path=str(out / "fail.png"))
            raise
        pg.add_style_tag(content=CSS)
        # A few plays of the user's and the opponent's turns, a chain if there is one, and the end.
        picks: list[tuple[str, int]] = []
        for want in (3, 4):
            ks = [k for k, f in enumerate(frames) if f["turn"] == want and f["caption"]]
            if ks:
                picks.append((f"watch-turn{want}", ks[len(ks) // 2]))
        chain = [k for k, f in enumerate(frames) if f["view"].get("chain")]
        if chain:
            picks.append(("watch-chain", chain[0]))
        cue = [k for k, f in enumerate(frames) if k > 0 and f["cues"] and f["turn"] >= 3]
        picks.append(("watch-end", len(frames) - 1))
        for name, i in picks:
            pg.evaluate("i => window.showFrame(i, {cues: false})", i)
            pg.evaluate("""([b, t]) => {
                document.getElementById('wtop')?.remove(); document.body.insertAdjacentHTML('beforeend', t);
                const s = document.getElementById('sheet') || document.getElementById('app');
                document.getElementById('wbar')?.remove(); s.insertAdjacentHTML('beforeend', b);
            }""", [bar(i), TOP])
            pg.wait_for_timeout(400)
            pg.screenshot(path=str(out / f"{name}.png"))
            print("shot", name, "frame", i, "turn", frames[i]["turn"], flush=True)
        if cue:
            # A cue scene frozen half-way: the live duel's animation, as the video will show it.
            i = cue[0]
            plan = pg.evaluate("i => window.cuePlan(i)", i)
            if plan:
                pg.evaluate("i => window.showFrame(i, {cue: 0, t: 0.5})", i)
                pg.evaluate("""([b, t]) => {
                    document.getElementById('wtop')?.remove(); document.body.insertAdjacentHTML('beforeend', t);
                    const s = document.getElementById('sheet') || document.getElementById('app');
                    document.getElementById('wbar')?.remove(); s.insertAdjacentHTML('beforeend', b);
                }""", [bar(i), TOP])
                pg.wait_for_timeout(300)
                pg.screenshot(path=str(out / "watch-cue.png"))
                print("shot watch-cue frame", i, flush=True)
        b.close()
finally:
    srv.terminate()
if errs:
    sys.exit("page errors: " + " | ".join(errs))
