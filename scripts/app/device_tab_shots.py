"""Ask's "On this device" tab in the real app against the real Rust server (mocks local-check / local-downloading).

Shows the tab with Gemma 4 E4B not downloaded, paused part-way and installed. The partial and full model files are
sparse (no disk used, nothing downloaded), and are removed at the end; a file already there is left alone.

    uv run --no-project --with playwright==1.57.0 python -u scripts/app/device_tab_shots.py <out_dir>
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
mdir = ROOT / "models/gemma-4-e4b"
full, part = mdir / "gemma-4-E4B-it.litertlm", mdir / "gemma-4-E4B-it.litertlm.part"
SIZE = 3_659_530_240
if full.exists() or part.exists():
    sys.exit(f"{mdir} already holds a model file: move it away first (this check fakes its states)")
ask_dir = ROOT / "data/cache/device_ui/ask"
shutil.rmtree(ask_dir, ignore_errors=True)
log = (out / "server.log").open("wb")
srv = subprocess.Popen([str(ROOT / "rust/target/release/decklight-serve"), "--port", "8813", "--token", "smoke",
                        "--ask-dir", str(ask_dir), "--records", str(ROOT / "data/cache/device_ui/records")],
                       cwd=ROOT, stdout=log, stderr=log)
errs: list[str] = []
problems: list[str] = []


def pick(pg, preset: str) -> None:
    # The provider is picked in its own searchable sheet (mock provider-pick), opened from the Model sheet's row.
    pg.click('.ak-prow'); pg.wait_for_selector("#akprovq")
    pg.click(f'[data-act="ask:ptab"][data-preset="{preset}"]'); pg.wait_for_selector(".ak-prow")


def sheet(pg, name: str, want: str) -> None:
    pick(pg, "device")
    pg.wait_for_selector(".ak-dcheck", timeout=10000)
    pg.wait_for_timeout(400)
    text = pg.inner_text(".ak-bsheet")
    if want not in text:
        problems.append(f"{name}: {want!r} not in the sheet: {text[:300]!r}")
    pg.screenshot(path=str(out / f"{name}.png"))
    print("shot", name, flush=True)


try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto("http://127.0.0.1:8813/?t=smoke&ltab=play")
        pg.wait_for_selector(".bbar svg", timeout=60000)
        pg.click('.bbar button[data-tab="ask"]')
        pg.wait_for_selector(".ak-empty")
        pg.click('[data-act="ask:model"]')
        pg.wait_for_selector(".ak-prow")
        sheet(pg, "1-none", "Download")
        mdir.mkdir(parents=True, exist_ok=True)
        with part.open("wb") as f:
            f.truncate(1_600_000_000)
        pick(pg, "opencode-go")
        sheet(pg, "2-paused", "Paused at 1.6 of 3.7 GB")
        part.unlink()
        with full.open("wb") as f:
            f.truncate(SIZE)
        pick(pg, "opencode-go")
        sheet(pg, "3-installed", "Remove")
        pg.click('.ak-pick[data-model="gemma-4-e4b"]')
        pg.wait_for_timeout(600)
        head = pg.inner_text(".ak-head") if pg.query_selector(".ak-head") else ""
        if "On this device" not in head and "gemma" not in head.lower():
            problems.append(f"picking Gemma did not reach the chat head: {head!r}")
        pg.screenshot(path=str(out / "4-picked.png"))
        b.close()
finally:
    srv.terminate()
    for f in (full, part):
        f.unlink(missing_ok=True)
    shutil.rmtree(ROOT / "data/cache/device_ui", ignore_errors=True)
print("page errors", errs)
print("problems", problems)
sys.exit(1 if errs or problems else 0)
