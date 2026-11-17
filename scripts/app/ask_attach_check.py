"""Ask's + attach (mocks ask-attach*): a card, a sample game and a photo go with a question, and the model gets them.
A fake OpenAI-compatible endpoint (custom preset) records the request: it must hold the photo as an image part and the
card and game as a tool round before the model's turn. Screenshots of each step go to <out_dir>.
    uv run --no-project --with playwright==1.57.0 python -u scripts/app/ask_attach_check.py <out_dir>"""
import json, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path.cwd(); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
PHOTO = next((ROOT / "data/cache/pics/full").glob("*.jpg"))
seen = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def reply(self, body, ctype="application/json"):
        b = body.encode(); self.send_response(200); self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self.reply(json.dumps({"data": [{"id": "fake"}]}))

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        seen.append(req)
        chunk = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "That card is [[Ash Blossom & Joyous Spring]]."}, "finish_reason": None}]}
        end = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        if req.get("stream"):
            self.reply("data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(end) + "\n\ndata: [DONE]\n\n", "text/event-stream")
        else:
            self.reply(json.dumps({"choices": [{"index": 0, "message": chunk["choices"][0]["delta"], "finish_reason": "stop"}],
                                   "usage": end["usage"]}))


fake = ThreadingHTTPServer(("127.0.0.1", 8819), Fake)
threading.Thread(target=fake.serve_forever, daemon=True).start()
(out / "ask").mkdir(exist_ok=True)
(out / "ask/settings.json").write_text(json.dumps({"preset": "custom", "model": "fake", "base_url": "http://127.0.0.1:8819/v1"}))
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8818", "--token", "smoke", "--ask-dir", str(out / "ask"),
                        "--records", str(out / "records")], stdout=open(out / "server.log", "wb"), stderr=subprocess.STDOUT)
problems = []
try:
    time.sleep(3)
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 412, "height": 915}, has_touch=True)
        pg.on("pageerror", lambda e: problems.append("page error: " + str(e)))
        pg.goto("http://127.0.0.1:8818/?t=smoke"); pg.wait_for_selector(".bbar svg", timeout=60000)
        pg.click('[data-tab="ask"]'); pg.wait_for_selector(".ak-plus"); pg.wait_for_timeout(1000)
        print("step: open the attach sheet", flush=True)
        pg.tap(".ak-plus"); pg.wait_for_selector(".ak-arow"); pg.wait_for_timeout(400); pg.screenshot(path=str(out / "1-sheet.png"))
        print("step: attach a card", flush=True)
        pg.tap('[data-act="ask:attach-card"]'); pg.fill("#akattq", "Ash Blossom"); pg.wait_for_selector('[data-act="ask:attach-pick"][data-kind="card"]', timeout=15000)
        pg.wait_for_timeout(300); pg.screenshot(path=str(out / "2-card.png"))
        pg.tap('[data-act="ask:attach-pick"][data-kind="card"]'); pg.wait_for_timeout(500)
        print("step: attach a sample game", flush=True)
        pg.tap(".ak-plus"); pg.tap('[data-act="ask:attach-game"]')
        pg.wait_for_selector('[data-act="ask:attach-pick"][data-kind="game"]', timeout=60000); pg.wait_for_timeout(300)
        pg.screenshot(path=str(out / "3-game.png"))
        pg.locator('[data-act="ask:attach-pick"][data-kind="game"]').first.tap(); pg.wait_for_timeout(500)
        print("step: attach a photo", flush=True)
        pg.tap(".ak-plus"); pg.tap('[data-act="ask:attach-photo"]'); pg.wait_for_timeout(300); pg.screenshot(path=str(out / "4-photo.png"))
        with pg.expect_file_chooser() as fc:
            pg.tap('[data-act="ask:photo-pick"]')
        fc.value.set_files(str(PHOTO))
        pg.wait_for_function("document.querySelectorAll('.ak-composer .ak-chip').length === 3", timeout=15000)
        pg.fill("#akin", "What does this card do against this deck?"); pg.wait_for_timeout(300)
        pg.screenshot(path=str(out / "5-composer.png"))
        print("step: send", flush=True)
        pg.tap(".ak-send"); pg.wait_for_selector(".ak-a", timeout=30000); pg.wait_for_timeout(1500)
        pg.screenshot(path=str(out / "6-sent.png"))
        chips = pg.locator(".ak-q .ak-chip").count()
        thumb = pg.locator(".ak-q .ak-chip i").count()
        if chips != 3: problems.append(f"the sent question shows {chips} chips, not 3")
        if thumb != 1: problems.append(f"the sent question shows {thumb} photo thumbnails, not 1")
        if pg.locator(".ak-composer .ak-chip").count(): problems.append("the composer kept its chips after sending")
        print("step: edit and ask again keeps the attachments", flush=True)
        n_before = len(seen)
        pg.tap(".ak-q"); pg.tap('[data-act="ask:editq"]'); pg.wait_for_timeout(400)
        kept = pg.locator(".ak-composer .ak-chip").count()
        if kept != 3: problems.append(f"editing the question kept {kept} chips, not 3")
        pg.fill("#akin", "And going second?"); pg.tap(".ak-send"); pg.wait_for_timeout(3000)
        pg.screenshot(path=str(out / "7-edited.png"))
        if len(seen) == n_before: problems.append("the edited question was not sent")
        print("step: a new chat starts without attachments", flush=True)
        pg.tap(".ak-plus"); pg.tap('[data-act="ask:attach-card"]'); pg.fill("#akattq", "Maxx")
        pg.wait_for_selector('[data-act="ask:attach-pick"][data-kind="card"]', timeout=15000)
        pg.locator('[data-act="ask:attach-pick"][data-kind="card"]').first.tap(); pg.wait_for_timeout(300)
        pg.tap('[data-act="ask:new"]'); pg.wait_for_timeout(300)
        if pg.locator(".ak-composer .ak-chip").count(): problems.append("a new chat kept the old chat's attachments")
        b.close()
finally:
    srv.kill(); fake.shutdown()

if not seen:
    problems.append("the fake model got no request")
for k, req in enumerate(seen):
    msgs = req["messages"]
    (out / f"request-{k}.json").write_text(json.dumps(req, indent=1)[:200000])
    # The photo is the last message (after the attached reads), so Gemma takes it as the question's image.
    parts = msgs[-1]["content"] if msgs[-1]["role"] == "user" else None
    if not isinstance(parts, list) or not any(x.get("type") == "image_url" and x["image_url"]["url"].startswith("data:image/jpeg;base64,") for x in parts):
        problems.append(f"request {k}: the last message is not the user's JPEG photo")
    # Only the newest question's attachments are read (an older one is named in its text).
    calls = [c["function"]["name"] for m in msgs if m["role"] == "assistant" for c in m.get("tool_calls") or []]
    tools = [m for m in msgs if m["role"] == "tool"]
    print("attached calls", calls, "results", [t["content"][:80] for t in tools], flush=True)
    if sorted(calls) != ["card", "replay"]: problems.append(f"request {k}: the attached tool round is {calls}, not card and replay")
    if any("error" in json.loads(t["content"]) for t in tools): problems.append("an attached read failed: " + str([t["content"][:200] for t in tools]))
    if any("attach" in m for m in msgs): problems.append("an `attach` key reached the provider")
print("problems", problems)
sys.exit(1 if problems else 0)
