"""Duel UI acceptance check (Playwright, < 1 min, mock mode).

Checks the phone fixes from the user's screenshots:
- tapping a hand card opens the centred action panel; every action button is
  fully inside the 360x780 viewport (no clipped labels); Cancel closes it and
  focus returns to the card; a prompt's own pass option replaces plain Cancel;
- Start duel with "Rock-paper-scissors" opens the RPS dialog, a draw asks
  again, a win asks first/second, and the duel then starts; Escape cancels
  without sending anything.

Run: uv run --with playwright python scripts/app/duel_ui_check.py
"""

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from web_screens import launch_kwargs, serve, menu_fits, INFO_UNDER_HAND_JS  # noqa: E402

ERR = []
# Indices in web/app.js MOCKS.
M04_SELECT = 3
M84_ACTIONS = 30
M85_ACTIONS_PASS = 31
M86_RPS_PICK = 32
M88_PROMPT_INFO = 34
M89_PAUSED = 35
M90_CUE_THEN_PROMPT = 36
M91_REVEAL_OVER_SHEET = 37
LOBBY = 0


def check(cond, msg):
    if not cond:
        ERR.append(msg)


VISIBLE_BOX_JS = """(sel) => {
  const el = document.querySelector(sel);
  if (!el) return null;
  const st = getComputedStyle(el);
  if (st.visibility === 'hidden' || Number(st.opacity) <= 0.05 || el.closest('[inert]')) return null;
  const r = el.getBoundingClientRect();
  if (r.width < 2 || r.height < 2) return null;
  return {l: r.left, t: r.top, r: r.right, b: r.bottom};
}"""

ACTIVE_BOX_JS = """() => {
  const k = window.__stageActive ? window.__stageActive() : null;
  const map = {sheet: '#sheet', menu: '#menu', reveal: '#reveal', detail: '#detail', rps: '#rps'};
  const sel = map[k];
  if (!sel) return {kind: k, mapped: false};
  const el = document.querySelector(sel);
  if (!el) return {kind: k, mapped: true, missing: true};
  const st = getComputedStyle(el);
  const r = el.getBoundingClientRect();
  return {kind: k, mapped: true, l: r.left, t: r.top, r: r.right, b: r.bottom,
          w: r.width, h: r.height, op: Number(st.opacity), vis: st.visibility,
          aside: el.classList.contains('stage-aside'), inert: !!el.closest('[inert]')};
}"""


def visible_box(page, sel):
    return page.evaluate(VISIBLE_BOX_JS, sel)


def overlaps(a, b):
    return bool(a and b and a["l"] < b["r"] - 0.5 and b["l"] < a["r"] - 0.5
                and a["t"] < b["b"] - 0.5 and b["t"] < a["b"] - 0.5)


def check_no_overlap(page, label, a, b):
    check(not overlaps(visible_box(page, a), visible_box(page, b)),
          f"{label}: {a} and {b} overlap")


def check_active_in_viewport(page, label, kind, width, height):
    box = page.evaluate(ACTIVE_BOX_JS)
    check(box.get("kind") == kind, f"{label}: active surface is {box.get('kind')}, expected {kind}")
    if not box.get("mapped") or box.get("missing"):
        check(False, f"{label}: no box for the active surface")
        return
    check(box["op"] > 0.05 and box["vis"] != "hidden" and not box["aside"] and not box["inert"],
          f"{label}: the active surface is dimmed or inert")
    check(box["l"] >= -0.5 and box["t"] >= -0.5 and box["r"] <= width + 0.5 and box["b"] <= height + 0.5,
          f"{label}: the active surface is outside the viewport ({box})")


def main():
    from playwright.sync_api import sync_playwright

    httpd, port = serve()
    base = f"http://127.0.0.1:{port}/index.html"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs())
            ctx = browser.new_context(viewport={"width": 360, "height": 780}, device_scale_factor=2, is_mobile=True)
            page = ctx.new_page()
            page.on("pageerror", lambda e: ERR.append(f"pageerror: {e}"))
            page.on(
                "console",
                lambda m: ERR.append(f"console-{m.type}: {m.text}") if m.type == "error" else None,
            )

            # 84: the action panel is centred and every button fits at 360x780.
            page.goto(f"{base}?mock=1&i={M84_ACTIONS}", wait_until="networkidle")
            page.wait_for_timeout(400)
            check(page.evaluate("!!document.querySelector('#menu[role=dialog][aria-modal=true]')"),
                  "hand card: no action dialog")
            check("Ash Blossom & Joyous Spring" in page.evaluate("document.querySelector('#menu').textContent"),
                  "hand card: no card name in the panel")
            check(page.evaluate("document.querySelectorAll('#menu .mactions .ctxbtn').length") >= 3,
                  "hand card: fewer than 3 action buttons")
            fit = menu_fits(page)
            check(fit == "ok", f"hand card: action panel clipped ({fit})")
            # Cancel closes it and returns focus to the card that opened it.
            page.locator('#menu [data-act="closemenu"]').first.click()
            page.wait_for_timeout(250)
            check(not page.evaluate("!!document.querySelector('#menu')"), "cancel did not close the action panel")
            card = page.locator('.card[data-uid="yh1"]').first
            check(card.count() == 1, "the hand card disappeared after cancel")
            card.click()
            page.wait_for_timeout(250)
            check(page.evaluate("!!document.querySelector('#menu')"), "tapping the hand card did not reopen the panel")
            check(page.evaluate("!!document.activeElement.closest('#menu')"), "focus did not move into the action panel")
            page.locator('#menu [data-act="closemenu"]').first.click()
            page.wait_for_timeout(250)
            check(page.evaluate("!document.querySelector('#menu')"), "cancel did not close the reopened panel")
            check(page.evaluate("document.activeElement && document.activeElement.getAttribute('data-uid')") == "yh1",
                  "focus did not return to the hand card")

            # 85: a prompt's own pass option is shown instead of a plain Cancel.
            page.goto(f"{base}?mock=1&i={M85_ACTIONS_PASS}", wait_until="networkidle")
            page.wait_for_timeout(400)
            text = page.evaluate("document.querySelector('#menu') ? document.querySelector('#menu').textContent : ''")
            check("Pass (do not respond)" in text, "pass prompt: no pass option in the panel")
            check(not page.evaluate("!!document.querySelector('#menu [data-act=closemenu]')"),
                  "pass prompt: a plain Cancel is shown beside the pass option")
            fit = menu_fits(page)
            check(fit == "ok", f"pass prompt: action panel clipped ({fit})")

            # 88: the prompt's subject card has an ⓘ and a tappable thumbnail, both open its details.
            page.goto(f"{base}?mock=1&i={M88_PROMPT_INFO}", wait_until="networkidle")
            page.wait_for_timeout(400)
            info = page.locator("#sheet .stitle .askinfo")
            check(info.count() == 1, "subject info: no ⓘ in the prompt header")
            check(info.get_attribute("aria-label") == "About Diabellstar the Black Witch",
                  "subject info: wrong aria-label on the header ⓘ")
            info.click()
            page.wait_for_timeout(250)
            check(page.evaluate("!!document.querySelector('#detail')"), "subject info: the header ⓘ did not open the detail")
            check("Diabellstar the Black Witch" in page.evaluate("document.querySelector('#detail').textContent"),
                  "subject info: the detail is not the prompt's card")
            page.keyboard.press("Escape")
            page.wait_for_timeout(250)
            page.locator("#sheet .stitle .askbtn").first.click()
            page.wait_for_timeout(250)
            check(page.evaluate("!!document.querySelector('#detail')"), "subject info: the header thumbnail did not open the detail")
            check("Diabellstar the Black Witch" in page.evaluate("document.querySelector('#detail').textContent"),
                  "subject info: the thumbnail detail is not the prompt's card")

            # 89: Pause holds incoming bot frames; Resume applies them in order. Space also toggles.
            page.goto(f"{base}?mock=1&i={M89_PAUSED}", wait_until="networkidle")
            page.wait_for_timeout(400)
            check(page.evaluate("!!document.querySelector('#pausebanner')"), "pause: no paused banner")
            check(page.evaluate("document.querySelector('.pausebtn')?.getAttribute('aria-pressed')") == "true",
                  "pause: the pause button is not pressed")
            before = page.evaluate("document.querySelector('.pbar.bot .lp').textContent")
            page.evaluate("""() => {
              const m = JSON.parse(JSON.stringify(window.__msg()));
              m.view.bot.lp = 7000;
              m.prompt = null; m.result = null; m.cues = []; m.log = [];
              window.__state(m);
            }""")
            page.wait_for_timeout(250)
            check(page.evaluate("document.querySelector('.pbar.bot .lp').textContent") == before,
                  "pause: a bot frame was applied while paused")
            check(page.evaluate("window.__pauseLen()") == 1, "pause: the incoming frame was not queued")
            page.locator(".pausebtn").click()
            page.wait_for_timeout(400)
            check(page.evaluate("window.__paused()") is False, "pause: Resume did not leave the paused state")
            check(not page.evaluate("!!document.querySelector('#pausebanner')"), "pause: the banner stayed after Resume")
            check(page.evaluate("document.querySelector('.pbar.bot .lp').textContent") == "7000",
                  "pause: the queued frame was not applied on Resume")
            check(page.evaluate("window.__pauseLen()") == 0, "pause: the queue did not drain")
            # Space toggles when focus is not in a text field.
            page.evaluate("document.activeElement && document.activeElement.blur()")
            page.keyboard.press("Space")
            page.wait_for_timeout(250)
            check(page.evaluate("window.__paused()") is True, "pause: Space did not pause")
            page.keyboard.press("Space")
            page.wait_for_timeout(250)
            check(page.evaluate("window.__paused()") is False, "pause: Space did not resume")

            # 90: a prompt that arrives during a cue waits; the sheet never covers the cue.
            page.goto(f"{base}?mock=1&i={M90_CUE_THEN_PROMPT}", wait_until="networkidle")
            page.wait_for_timeout(250)
            check(page.evaluate("!!document.querySelector('#ov-cues .cue-act')"), "prompt wait: no cue on screen")
            check(not page.evaluate("!!document.querySelector('#sheet')"), "prompt wait: the sheet opened over the cue")
            overlap = False
            appeared = False
            for _ in range(60):
                if page.evaluate("!!document.querySelector('#ov-cues .cue-act') && !!document.querySelector('#sheet')"):
                    overlap = True
                    break
                if page.evaluate("!!document.querySelector('#sheet')"):
                    appeared = True
                    break
                page.wait_for_timeout(100)
            check(not overlap, "prompt wait: the sheet covered the cue")
            check(appeared, "prompt wait: the prompt never appeared")
            check(page.evaluate("!!document.querySelector('#sheet .stitle .askinfo')"),
                  "prompt wait: the prompt header lost its ⓘ")

            # --- attention rules (docs/design/ui-attention.md) ---
            # Cue vs card actions: the user-opened panel goes first; the cue waits
            # behind it (hidden, not peeking around the sheet).
            page.goto(f"{base}?mock=1&i={M90_CUE_THEN_PROMPT}&cue=0&t=0.7&menu=yh1", wait_until="networkidle")
            page.wait_for_timeout(400)
            check(page.evaluate("window.__stageActive()") == "cues",
                  "cue vs actions: the cue is not the active narration")
            check(not visible_box(page, "#menu"), "cue vs actions: the card action panel is visible over the cue")
            check_no_overlap(page, "cue vs actions", "#ov-cues .cue-act", "#menu")

            # Game menu vs cue: a lookup the user opened stays; narration queues.
            page.goto(f"{base}?mock=1&i={M90_CUE_THEN_PROMPT}&cue=0&t=0.7&gamemenu=1", wait_until="networkidle")
            page.wait_for_timeout(400)
            check(page.evaluate("window.__stageActive()") == "gamemenu",
                  "game menu vs cue: the game menu is not the active lookup")
            check(visible_box(page, "#gamemenu"), "game menu vs cue: no game menu")
            check(not page.evaluate("(() => { const e = document.querySelector('#ov-cues');"
                                    " return !!e && getComputedStyle(e).visibility !== 'hidden'; })()"),
                  "game menu vs cue: the cue layer is still visible")
            check_no_overlap(page, "game menu vs cue", "#ov-cues .cue-act", "#gamemenu")

            # Reveal vs sheet: the reveal takes the stage, the select sheet steps
            # fully aside (still in the DOM, no dimmed sliver), then returns.
            page.goto(f"{base}?mock=1&i={M91_REVEAL_OVER_SHEET}", wait_until="networkidle")
            page.wait_for_timeout(450)
            check(page.evaluate("window.__stageActive()") == "reveal",
                  "reveal vs sheet: the reveal is not the active surface")
            check(visible_box(page, "#reveal"), "reveal vs sheet: no reveal on screen")
            check(not visible_box(page, "#sheet"), "reveal vs sheet: the prompt sheet is visible under the reveal")
            check_no_overlap(page, "reveal vs sheet", "#reveal", "#sheet")
            rface = page.evaluate("""() => { const el = document.querySelector('#reveal .rface');
              if (!el) return null; const r = el.getBoundingClientRect();
              return {l: r.left, t: r.top, r: r.right, b: r.bottom}; }""")
            check(bool(rface) and rface["l"] >= -0.5 and rface["t"] >= -0.5
                  and rface["r"] <= 360.5 and rface["b"] <= 780.5,
                  f"reveal vs sheet: the revealed card is clipped ({rface})")
            # A landed-card ghost (board patch) lies under the reveal, never on it (user 2026-11-12: "the card that
            # goes to the field stays on the top of the text/revealed stuff").
            top = page.evaluate("""() => { const host = document.getElementById('cue-ghosts');
              const t = document.querySelector('#reveal .rtext') || document.querySelector('#reveal h2');
              if (!host || !t) return 'missing ' + (host ? 'reveal text' : '#cue-ghosts');
              const r = t.getBoundingClientRect(); const g = document.createElement('div');
              g.className = 'cue-card ghost'; g.style.cssText = 'position:absolute;left:' + r.left + 'px;top:' + r.top +
                'px;width:' + r.width + 'px;height:' + r.height + 'px;background:#fff';
              host.appendChild(g); const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
              g.remove(); return hit && hit.closest('#reveal') ? 'reveal' : (hit ? (hit.id || hit.className) : 'none'); }""")
            check(top == "reveal", f"ghost vs reveal: a landed-card ghost is drawn over the reveal text ({top})")
            page.locator("#reveal").click(position={"x": 8, "y": 8})
            page.wait_for_timeout(350)
            check(page.evaluate("window.__stageActive()") == "sheet",
                  "reveal vs sheet: the decision did not come back after the reveal")
            check(visible_box(page, "#sheet"), "reveal vs sheet: the sheet did not return")
            check(not page.evaluate("!!document.querySelector('#reveal')"), "reveal vs sheet: the reveal stayed")

            # Detail vs sheet: the lookup takes the stage; Escape returns the sheet.
            page.goto(f"{base}?mock=1&i={M88_PROMPT_INFO}", wait_until="networkidle")
            page.wait_for_timeout(400)
            check(page.evaluate("window.__stageActive()") == "sheet", "detail vs sheet: the sheet is not active")
            page.locator("#sheet .stitle .askinfo").click()
            page.wait_for_timeout(300)
            check(page.evaluate("window.__stageActive()") == "detail",
                  "detail vs sheet: the detail is not the active lookup")
            check(not visible_box(page, "#sheet"), "detail vs sheet: the sheet is visible under the detail")
            check_no_overlap(page, "detail vs sheet", "#detail", "#sheet")
            page.keyboard.press("Escape")
            page.wait_for_timeout(300)
            check(page.evaluate("window.__stageActive()") == "sheet",
                  "detail vs sheet: the sheet did not return after Escape")
            check(visible_box(page, "#sheet"), "detail vs sheet: no sheet after Escape")

            # Recap vs sheet: the decision keeps the stage; a passive recap waits.
            page.goto(f"{base}?mock=1&i={M04_SELECT}", wait_until="networkidle")
            page.wait_for_timeout(300)
            page.evaluate("""() => window.__recap({turn: 2, who: 'decklight', items: [
              {text: 'Activated a card', sub: '', card: null},
              {text: 'Set a card', sub: '', card: null},
              {text: 'Summoned a monster', sub: '', card: null}]})""")
            page.wait_for_timeout(250)
            check(page.evaluate("window.__stageActive()") == "sheet",
                  "recap vs sheet: the recap took the stage from the decision")
            check(not visible_box(page, "#info .recap"), "recap vs sheet: the recap is visible beside the sheet")
            check_no_overlap(page, "recap vs sheet", "#info .recap", "#sheet")
            page.evaluate("window.__recap(null)")
            page.wait_for_timeout(200)

            # State restored after stepping aside: selection, scroll and focus.
            page.goto(f"{base}?mock=1&i=2&sel=0", wait_until="networkidle")
            page.wait_for_timeout(400)
            page.evaluate("document.querySelector('#sheet .sbody').scrollTop = 40")
            page.locator('#sheet [data-opt="0"]').focus()
            before_scroll = page.evaluate("document.querySelector('#sheet .sbody').scrollTop")
            check(page.evaluate("document.querySelectorAll('#sheet .opt.sel').length") == 1,
                  "restore: the option was not selected before stepping aside")
            check(page.evaluate("!!window.__revealCard('bm1')"), "restore: could not raise a reveal")
            page.wait_for_timeout(250)
            check(not visible_box(page, "#sheet"), "restore: the sheet stayed visible during the reveal")
            page.locator("#reveal").click(position={"x": 8, "y": 8})
            page.wait_for_timeout(350)
            check(visible_box(page, "#sheet"), "restore: the sheet did not come back")
            check(page.evaluate("document.querySelectorAll('#sheet .opt.sel').length") == 1,
                  "restore: the selection was lost")
            check(page.evaluate("document.querySelector('#sheet .sbody').scrollTop") == before_scroll,
                  "restore: the sheet scroll position was lost")
            check(page.evaluate("document.activeElement && document.activeElement.getAttribute('data-opt')") == "0",
                  "restore: focus did not return to the same option")

            # Pause holds the decision; Resume brings it back unchanged.
            page.goto(f"{base}?mock=1&i={M04_SELECT}", wait_until="networkidle")
            page.wait_for_timeout(300)
            check(visible_box(page, "#sheet"), "pause stage: no sheet before pausing")
            page.locator(".pausebtn").click()
            page.wait_for_timeout(250)
            check(page.evaluate("window.__paused()") is True, "pause stage: Pause did not engage")
            check(not visible_box(page, "#sheet"), "pause stage: the sheet stayed up while paused")
            page.locator(".pausebtn").click()
            page.wait_for_timeout(300)
            check(page.evaluate("window.__paused()") is False, "pause stage: Resume did not engage")
            check(visible_box(page, "#sheet"), "pause stage: the sheet did not return after Resume")

            # Pause drains its frames in order.
            page.goto(f"{base}?mock=1&i=1", wait_until="networkidle")
            page.wait_for_timeout(350)
            page.locator(".pausebtn").click()
            page.wait_for_timeout(200)
            for lp in (7000, 6500):
                page.evaluate("""(lp) => {
                  const m = JSON.parse(JSON.stringify(window.__msg()));
                  m.view.bot.lp = lp; m.prompt = null; m.result = null; m.cues = []; m.log = [];
                  window.__state(m);
                }""", lp)
            page.wait_for_timeout(150)
            check(page.evaluate("window.__pauseLen()") == 2, "pause order: frames were not queued")
            page.locator(".pausebtn").click()
            page.wait_for_timeout(400)
            check(page.evaluate("window.__pauseLen()") == 0, "pause order: the queue did not drain")
            check(page.evaluate("document.querySelector('.pbar.bot .lp').textContent") == "6500",
                  "pause order: the last frame did not win")

            # Queue drains after narration: the prompt arrives after the cue, not on top of it.
            page.goto(f"{base}?mock=1&i={M90_CUE_THEN_PROMPT}", wait_until="networkidle")
            appeared = False
            for _ in range(60):
                if visible_box(page, "#sheet"):
                    appeared = True
                    break
                page.wait_for_timeout(100)
            check(appeared, "queue drain: the prompt never appeared within the fallback")
            check(not visible_box(page, "#ov-cues .cue-act"),
                  "queue drain: the cue and the prompt share the stage")
            check(page.evaluate("window.__stageActive()") == "sheet",
                  "queue drain: the sheet is not the active surface after the cue")

            # A long bot turn with a prompt behind it plays every scene, not all at once after a ceiling
            # (user 2026-11-12: "10 moves happened almost all at once by the bot").
            page.goto(f"{base}?mock=1&i=1", wait_until="networkidle")
            page.wait_for_timeout(350)
            m90 = json.loads((SCRIPTS.parent / "web" / "mock" / "90-cue-then-prompt.json").read_text())
            page.evaluate("""(m90) => {
              for (let k = 0; k < 6; k++) {
                const m = JSON.parse(JSON.stringify(m90));
                m.cues.forEach((c) => { c.n = (c.n || 0) + 1000 * (k + 1); });
                window.__state(m);
              }
            }""", m90)
            page.wait_for_timeout(6500)
            check(page.evaluate("window.__cue.busy()"),
                  "long turn: the bot's queued scenes were flushed at once while a prompt waited")

            # Info lines stay above the hand at real phone heights (S23 minus system bars ~ 412x800-860), with and
            # without the win-chance row (user 2026-11-12: "When I add the win chance the status moves ... hidden").
            for vw, vh in ((412, 800), (412, 840), (412, 860), (360, 740)):
                page.set_viewport_size({"width": vw, "height": vh})
                for q in ("1", "18&winchance=1", "15&winchance=1"):
                    page.goto(f"{base}?mock=1&i={q}", wait_until="networkidle")
                    page.wait_for_timeout(300)
                    if page.evaluate("document.documentElement.classList.contains('short')"):
                        continue
                    bad = page.evaluate(INFO_UNDER_HAND_JS)
                    check(not bad, f"info under hand at {vw}x{vh} mock {q}: {bad}")
            page.set_viewport_size({"width": 360, "height": 780})

            # Rock-paper-scissors: pick, draw asks again, win asks first/second, then the duel starts.
            page.goto(f"{base}?mock=1&i={LOBBY}", wait_until="networkidle")
            page.wait_for_timeout(400)
            page.locator("#startbtn").click()
            page.wait_for_timeout(250)
            check(page.evaluate("!!document.querySelector('#rps[role=dialog][aria-modal=true]')"),
                  "start: no rock-paper-scissors dialog")
            check(page.evaluate("document.querySelectorAll('#rps [data-act=rpspick]').length") == 3,
                  "start: need three hand buttons")
            check(page.evaluate("!!document.activeElement.closest('#rps')"),
                  "start: focus did not move into the RPS dialog")
            page.evaluate("window.__rpsSetHand(2)")
            page.locator('#rps [data-act="rpspick"][data-hand="2"]').click()
            page.wait_for_timeout(250)
            check("Draw" in page.evaluate("document.querySelector('#rps').textContent"),
                  "draw: the result is not shown")
            check(page.evaluate("!!document.querySelector('#rps [data-act=rpsagain]')"),
                  "draw: no throw-again button")
            page.evaluate("window.__rpsSetHand(1)")
            page.locator('#rps [data-act="rpsagain"]').click()
            page.wait_for_timeout(250)
            page.locator('#rps [data-act="rpspick"][data-hand="2"]').click()
            page.wait_for_timeout(250)
            text = page.evaluate("document.querySelector('#rps').textContent")
            check("You win" in text, "win: the win result is not shown")
            check("Go first" in text and "Go second" in text, "win: no first/second choices")
            page.locator('#rps [data-act="rpsfirst"][data-first="me"]').click()
            page.wait_for_selector("#app .field", timeout=5000)
            page.wait_for_timeout(250)
            check(not page.evaluate("!!document.querySelector('#rps')"), "the RPS dialog stayed open after choosing first")
            check("Your move" in page.evaluate("document.body.innerText") or page.evaluate("!!document.querySelector('#app .field')"),
                  "the duel did not start after choosing first")

            # Escape cancels the RPS and sends nothing (still on the lobby).
            page.goto(f"{base}?mock=1&i={LOBBY}", wait_until="networkidle")
            page.wait_for_timeout(400)
            page.locator("#startbtn").click()
            page.wait_for_timeout(250)
            check(page.evaluate("!!document.querySelector('#rps')"), "cancel: no RPS dialog to cancel")
            page.keyboard.press("Escape")
            page.wait_for_timeout(250)
            check(not page.evaluate("!!document.querySelector('#rps')"), "Escape did not close the RPS dialog")
            check(page.evaluate("!!document.querySelector('#lobby')"), "Escape did not return to the lobby")
            check(page.evaluate("!!document.querySelector('#startbtn:not([disabled])')"),
                  "the lobby Start button did not come back")

            # The active surface fits the whole viewport at both phone sizes.
            for vw, vh in ((360, 780), (412, 915)):
                if vw == 360:
                    vpage, vctx = page, None
                else:
                    vctx = browser.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=2, is_mobile=True)
                    vpage = vctx.new_page()
                for query, kind, click in (
                    (M04_SELECT, "sheet", None),
                    (M84_ACTIONS, "menu", None),
                    (M91_REVEAL_OVER_SHEET, "reveal", None),
                    (M88_PROMPT_INFO, "detail", "#sheet .stitle .askinfo"),
                    (M86_RPS_PICK, "rps", None),
                ):
                    vpage.goto(f"{base}?mock=1&i={query}", wait_until="networkidle")
                    vpage.wait_for_timeout(350)
                    if click:
                        vpage.locator(click).click()
                        vpage.wait_for_timeout(250)
                    check_active_in_viewport(vpage, f"{kind}@{vw}x{vh}", kind, vw, vh)
                if vctx:
                    vctx.close()

            ctx.close()
            browser.close()
    finally:
        httpd.shutdown()

    if ERR:
        print("FAIL:")
        for e in ERR:
            print("  " + e)
        return 1
    print("ok: card action panel fits, Cancel returns focus, RPS draw/win/cancel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
