"""Settings screen acceptance check (Playwright, < 1 min).

Opens Settings from the lobby gear, toggles Show win chance, closes with Escape and
checks focus returns to the gear. Then opens a duel's game menu and checks the switch
shows the new value, opens Settings from "All settings…", closes with Escape and checks
focus returns to the game-menu button.

Run: uv run --with playwright python scripts/app/settings_check.py
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from web_screens import launch_kwargs, serve  # noqa: E402

ERR = []


def check(cond, msg):
    if not cond:
        ERR.append(msg)


def main():
    from playwright.sync_api import sync_playwright

    httpd, port = serve()
    base = f"http://127.0.0.1:{port}/index.html"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs())
            ctx = browser.new_context(viewport={"width": 412, "height": 915}, device_scale_factor=2, is_mobile=True)
            page = ctx.new_page()
            page.on("pageerror", lambda e: ERR.append(f"pageerror: {e}"))
            page.on(
                "console",
                lambda m: ERR.append(f"console-{m.type}: {m.text}") if m.type == "error" else None,
            )

            # From the lobby: gear -> Settings -> Show win chance -> Escape -> focus back on the gear.
            page.goto(f"{base}?mock=1&i=27", wait_until="networkidle")
            page.wait_for_timeout(400)
            check(page.evaluate("!!document.querySelector('.lhead [data-act=opensettings]')"), "no gear in the lobby header")
            page.locator(".lhead [data-act=opensettings]").click()
            page.wait_for_timeout(300)
            check(page.evaluate("!!document.querySelector('#settings')"), "Settings did not open from the lobby gear")
            check(
                page.evaluate("document.activeElement && document.activeElement.id === 'settings-title'"),
                "focus did not move to the Settings heading",
            )
            before = page.evaluate(
                "document.querySelector('#settings [data-act=winchanceset]').getAttribute('aria-checked')"
            )
            page.locator("#settings [data-act=winchanceset]").click()
            page.wait_for_timeout(300)
            after = page.evaluate(
                "document.querySelector('#settings [data-act=winchanceset]').getAttribute('aria-checked')"
            )
            check(before != after and after == "true", f"Show win chance did not turn on ({before} -> {after})")
            check(
                page.evaluate("localStorage.getItem('ygo-winchance') === '1'"),
                "ygo-winchance was not saved",
            )
            page.keyboard.press("Escape")
            page.wait_for_timeout(400)
            check(not page.evaluate("!!document.querySelector('#settings')"), "Escape did not close Settings")
            check(
                page.evaluate("document.activeElement && document.activeElement.getAttribute('data-act') === 'opensettings'"),
                "focus did not return to the lobby gear",
            )

            # Same browser state, a duel: the game menu shows the new value, and Settings opened from
            # "All settings…" closes back to the duel with focus on the game-menu button.
            page.goto(f"{base}?mock=1&i=29&gamemenu=1", wait_until="networkidle")
            page.wait_for_timeout(400)
            val = page.evaluate(
                "document.querySelector('#gamemenu [data-act=winchanceset]')?.getAttribute('aria-checked')"
            )
            check(val == "true", f"game menu shows the old win-chance value ({val})")
            page.locator("#gamemenu [data-act=opensettings]").click()
            page.wait_for_timeout(300)
            check(page.evaluate("!!document.querySelector('#settings')"), "Settings did not open from the game menu")
            check(not page.evaluate("!!document.querySelector('#gamemenu')"), "the game menu stayed open under Settings")
            check(page.evaluate("!!document.querySelector('#app .field')"), "the duel board disappeared")
            page.keyboard.press("Escape")
            page.wait_for_timeout(400)
            check(not page.evaluate("!!document.querySelector('#settings')"), "Escape did not close Settings over the duel")
            check(
                page.evaluate("document.activeElement && document.activeElement.getAttribute('data-act') === 'gamemenu'"),
                "focus did not return to the game-menu button",
            )

            ctx.close()
            browser.close()
    finally:
        httpd.shutdown()

    if ERR:
        print("FAIL:")
        for e in ERR:
            print("  " + e)
        return 1
    print("ok: settings open/close, cross-screen value, focus return")
    return 0


if __name__ == "__main__":
    sys.exit(main())
