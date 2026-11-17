"""Ask + Play tab mock screenshots (docs/ui/mocks/ask/mocks.html), phone size.
Run: uv run --no-project --with playwright==1.57.0 python -u scripts/app/ask_mock_shots.py <out_dir> [screen,screen...]"""
import functools, http.server, sys, threading
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT)); h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8799), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
SHOTS = [(v, b) for b in ("top", "bottom") for v in ("play-bot", "play-people", "ask-answer")] + [
    (v, "bottom") for v in ("ask-empty", "ask-nokey", "ask-thinking", "ask-running", "tools-all", "ask-tool-open", "ask-limit", "ask-key-bad", "ask-list", "ask-settings",
    "icons", "model-go", "model-local", "ask-gemma", "chats", "fork-menu", "fork-edit", "fork-branches",
    "ask-replay", "ask-botview", "ask-sequence", "sequence-frames",
    "games", "games-empty", "game", "game-decision", "deck-odds",
    "ask-attach", "ask-attach-game", "ask-attached", "local-check", "local-downloading", "local-nogpu",
    "games-search", "games-select", "games-delete", "sequence-hand", "sequence-moves",
    "export-sheet", "export-range", "export-moment", "export-making", "export-ready", "export-failed", "export-many", "export-many-sheet", "export-frame", "model-provider", "provider-pick")]
if len(sys.argv) > 2:
    SHOTS = [s for s in SHOTS if s[0] in sys.argv[2].split(",")]
errs = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for i, (v, bar) in enumerate(SHOTS, 1):
        pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(f"http://127.0.0.1:8799/docs/ui/mocks/ask/mocks.html?v={v}&bar={bar}"); pg.wait_for_timeout(500)
        pg.screenshot(path=str(out / f"{i:02d}-{v}-{bar}.png")); pg.close()
        print(f"{i}/{len(SHOTS)} {v} {bar}", flush=True)
    b.close()
print("errors", errs)
sys.exit(1 if errs else 0)
