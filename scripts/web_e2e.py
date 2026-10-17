"""Browser end-to-end check of the web client against the real server.

Starts scripts/web_play.py, opens the page in Playwright Chromium at phone size, starts a duel from the lobby and
plays it by tapping like a person would (glowing cards, menu/sheet options, phase buttons) until the game-over modal.
Saves screenshots to data/cache/web-e2e/. Exits non-zero on a page error, a stall, or no game over.

Along the way (random mode) it checks the phase-3 UI behavior:
  (a) while the bot is thinking, the log drawer survives re-renders (same
      DOM node) and its X closes it;
  (b) tapping a face-up non-actionable field monster opens the card detail,
      and the detail X closes it;
  (c) a sheet info button opens the detail without choosing, and the prompt
      is still there afterwards.
After game over it starts a second game, surrenders through the in-game menu
and expects the game-over modal. With --surrender-thinking the second game
uses mode fast and surrenders while the bot is thinking.

  uv run --with playwright python scripts/web_e2e.py [--mode random|fast|product] [--max-s 240]
"""
from __future__ import annotations

import argparse
import random
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from web_screens import launch_kwargs

OUT = ROOT / "data" / "cache" / "web-e2e"


def modal_visible(page) -> bool:
    return page.locator("[data-act=rematch]").count() > 0


def tap(page, loc, timeout: int = 2000) -> bool:
    """Click like a person: a bot-activation reveal covers the screen while it
    shows, and a tap on it dismisses it first (then the tap is retried)."""
    try:
        loc.tap(timeout=timeout)
        return True
    except Exception:  # noqa: BLE001, S110
        pass
    for _ in range(4):  # a bot combo can queue a couple of reveals
        if not page.locator("#reveal").count():
            return False
        try:
            page.locator("#reveal").tap(timeout=1000)
        except Exception:  # noqa: BLE001, S110
            pass
        page.wait_for_timeout(400)
        try:
            loc.tap(timeout=timeout)
            return True
        except Exception:  # noqa: BLE001, S112
            continue
    return False


def check_a_log_drawer(page, errors: list[str], ran: dict) -> None:
    """(a) Log drawer keeps its DOM node across re-renders while the bot plays.

    The brief keys this on the thinking indicator, but the server only reports
    thinking in driver (product/fast) mode -- in random mode it is never set
    (session.py sets thinking_since only around driver.next_bot). So in random
    mode the trigger is the bot's turn being active (states stream in either
    way, which is what the check is about).
    """
    if ran["a"] or modal_visible(page):
        return
    thinking = page.locator(".pbar.bot .think").count() > 0
    bot_turn = page.locator(".phase.bot").count() > 0
    if not thinking and not bot_turn:
        return
    logbtns = [b for b in page.locator("[data-act=log]").all() if b.is_visible()]
    if not logbtns:
        return
    try:
        if not tap(page, logbtns[0]):
            return  # drawer did not open; try again later
        try:
            page.wait_for_selector("#logdrawer", timeout=2000)
        except Exception:  # noqa: BLE001
            errors.append("(a) the log button did not open the drawer within 2 s")
            return
        ran["a"] = True
        # Mark the live node with a JS expando (morph keeps the node itself;
        # attributes are synced from the template, so dataset would not survive).
        page.evaluate("document.querySelector('#logdrawer').__e2e = 1")
        page.wait_for_timeout(1500)
        same = page.evaluate(
            "!!(document.querySelector('#logdrawer') && "
            "document.querySelector('#logdrawer').__e2e === 1)")
        if not same:
            errors.append("(a) log drawer node was recreated while the bot played")
            return
        if modal_visible(page):
            return  # game ended mid-check; the stability assertion already passed
        if not tap(page, page.locator("#logdrawer [data-act=closelog]").first, timeout=3000):
            errors.append("(a) log drawer X did not close the drawer")
            return
        page.wait_for_timeout(300)
        if page.locator("#logdrawer").count() != 0:
            errors.append("(a) log drawer X did not close the drawer")
    except Exception as e:  # noqa: BLE001
        errors.append(f"(a) log drawer check failed: {e}")


def check_b_field_detail(page, errors: list[str], ran: dict) -> None:
    """(b) Tapping a field monster opens the detail overlay."""
    if ran["b"] or modal_visible(page):
        return
    cards = [c for c in page.locator(
        ".field [data-row=mzone] .slot:not(.side) .card[data-uid]:not(.actionable):not(.back)").all() if c.is_visible()]
    if not cards:
        return
    try:
        if not tap(page, cards[0]):
            return  # blocked (reveal/state change); try again later
        page.wait_for_timeout(300)
        ran["b"] = True
        if page.locator("#detail").count() == 0:
            errors.append("(b) tapping a field monster did not open the detail")
            return
        if not tap(page, page.locator("#detail .dx"), timeout=3000):
            errors.append("(b) detail X did not close the detail")
            return
        page.wait_for_timeout(300)
        if page.locator("#detail").count() != 0:
            errors.append("(b) detail X did not close the detail")
    except Exception as e:  # noqa: BLE001
        errors.append(f"(b) field detail check failed: {e}")


def check_c_sheet_info(page, errors: list[str], ran: dict) -> None:
    """(c) A sheet info button opens the detail without choosing."""
    if ran["c"] or modal_visible(page):
        return
    infos = [b for b in page.locator("#sheet .infobtn").all() if b.is_visible()]
    if not infos:
        return
    try:
        if not tap(page, infos[0]):
            return  # blocked (reveal/state change); try again later
        page.wait_for_timeout(300)
        ran["c"] = True
        if page.locator("#detail").count() == 0:
            errors.append("(c) sheet info button did not open the detail")
            return
        if page.locator("#sheet").count() == 0:
            errors.append("(c) prompt sheet went away after tapping the info button")
        if not tap(page, page.locator("#detail .dx"), timeout=3000):
            errors.append("(c) detail X did not close the detail")
            return
        page.wait_for_timeout(300)
        if page.locator("#detail").count() != 0:
            errors.append("(c) detail X did not close the detail")
    except Exception as e:  # noqa: BLE001
        errors.append(f"(c) sheet info check failed: {e}")


def check_d_closers(page, errors: list[str], ran: dict) -> None:
    """(d) Pile sheet Close, pile scrim, log scrim and game-menu X all close (touch taps)."""
    if ran["d"] or modal_visible(page) or page.locator("#sheet:not(.pile)").count():
        return
    piles = [p for p in page.locator("[data-pile]").all() if p.is_visible()]
    if not piles:
        return
    ran["d"] = True

    def closes(label, open_loc, sel, close_loc):
        if not tap(page, open_loc, timeout=3000):
            errors.append(f"(d) {label}: could not open")
            return
        page.wait_for_timeout(300)
        if page.locator(sel).count() == 0:
            errors.append(f"(d) {label}: did not open")
            return
        cl = close_loc()
        if isinstance(cl, tuple):  # a scrim: tap its uncovered top-left strip
            try:
                cl[0].tap(position={"x": 16, "y": 40}, timeout=3000)
                ok = True
            except Exception:  # noqa: BLE001
                ok = False
        else:
            ok = tap(page, cl, timeout=3000)
        if not ok:
            errors.append(f"(d) {label}: close tap failed")
            return
        page.wait_for_timeout(400)
        if page.locator(sel).count() != 0:
            errors.append(f"(d) {label}: still open after close")

    try:
        closes("pile Close", piles[0], "#sheet.pilesheet", lambda: page.locator("[data-act=closepile].bigbtn"))
        closes("pile scrim", piles[0], "#sheet.pilesheet",
               lambda: (page.locator("#pscrim"),))
        logb = page.locator("[data-act=log]").first
        closes("log scrim", logb, "#logdrawer", lambda: (page.locator("#scrim"),))
        # Android back gesture: closes the drawer, and only the drawer
        if tap(page, page.locator("[data-act=log]").first, timeout=3000):
            page.wait_for_timeout(300)
            page.evaluate("history.back()")
            page.wait_for_timeout(500)
            if page.locator("#logdrawer").count():
                errors.append("(d) back gesture did not close the log drawer")
            elif page.locator("#app").count() == 0:
                errors.append("(d) back gesture left the page")
        closes("game menu X", page.locator("[data-act=gamemenu]").first, "#gamemenu",
               lambda: page.locator("#gamemenu [data-act=closegamemenu]").first)
    except Exception as e:  # noqa: BLE001
        errors.append(f"(d) closers check failed: {e}")


_OVERLAYS = [("#detail", "#detail .dx"), ("#logdrawer", "#logdrawer [data-act=closelog]"),
             ("#gamemenu", "#gamemenu [data-act=closegamemenu]"),
             ("#sheet.pilesheet", "[data-act=closepile].bigbtn")]


def close_stray_overlays(page, errors: list[str]) -> None:
    """An overlay left open (a random tap landed on a card that stopped being
    actionable, ...) is closed with its X like a person would; one that stays
    open after the tap is a bug: fail loudly with which one."""
    if modal_visible(page):
        return
    for sel, close in _OVERLAYS:
        if page.locator(sel).count() == 0:
            continue
        ok = tap(page, page.locator(close).first, timeout=3000)
        page.wait_for_timeout(400)
        if not ok or page.locator(sel).count():
            info = page.evaluate("""() => ({ov: history.state, top: (() => { const b = document.querySelector('%s');
                if (!b) return 'no close control'; const r = b.getBoundingClientRect();
                const e = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
                return e ? (e.id || e.className || e.tagName) : null; })()})""" % close)  # noqa: UP031 -- a JS template, braces clash with format()
            errors.append(f"overlay {sel} stayed open after tapping {close} (tap ok={ok}, {info})")
            return


def surrender_through_menu(page, errors: list[str], within_s: float, label: str) -> bool:
    """Open the in-game menu and tap Surrender twice; wait for game over."""
    try:
        if not tap(page, page.locator("[data-act=gamemenu]").first, timeout=5000):
            errors.append(f"{label}: game menu did not open")
            return False
        page.wait_for_timeout(300)
        if page.locator("#gamemenu").count() == 0:
            errors.append(f"{label}: game menu did not open")
            return False
        if not tap(page, page.locator("#gamemenu [data-act=surrender]"), timeout=3000):
            errors.append(f"{label}: surrender tap failed")
            return False
        page.wait_for_timeout(300)
        txt = page.locator("#gamemenu [data-act=surrender]").first.inner_text()
        if "again" not in txt:
            errors.append(f"{label}: surrender did not ask for confirmation (got {txt!r})")
        if not tap(page, page.locator("#gamemenu [data-act=surrender]"), timeout=3000):
            errors.append(f"{label}: surrender confirm tap failed")
            return False
        t0 = time.time()
        while time.time() - t0 < within_s:
            if modal_visible(page):
                return True
            page.wait_for_timeout(200)
        errors.append(f"{label}: no game-over modal within {within_s:.0f} s of surrender")
        return False
    except Exception as e:  # noqa: BLE001
        errors.append(f"{label}: surrender flow failed: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="random", choices=["random", "fast", "product"])
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--max-s", type=float, default=240)
    ap.add_argument("--shot-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--surrender-thinking", action="store_true",
                    help="second game uses mode fast and surrenders while the bot thinks")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    rng = random.Random(args.seed)
    tok = secrets.token_hex(8)
    url = f"http://127.0.0.1:{args.port}/?t={tok}"
    srv = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "web_play.py"), "--port", str(args.port),
                            "--token", tok], cwd=ROOT, stdout=open(OUT / "server.log", "w"),  # noqa: SIM115 -- the server lives for the whole run
                           stderr=subprocess.STDOUT)
    errors: list[str] = []
    ran = {"a": False, "b": False, "c": False, "d": False}
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        else:
            print("server did not start; see", OUT / "server.log")
            return 1
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs())
            page = browser.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2,
                                    is_mobile=True, has_touch=True)  # Chrome Android: taps
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e} | {getattr(e, 'stack', '')}"))
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                    if m.type == "error" and "404" not in m.text else None)
            frames: list[str] = []
            page.on("websocket", lambda ws: ws.on("framereceived", lambda f: frames.append(f)))
            page.goto(url)
            page.wait_for_selector("#startbtn", timeout=10000)
            page.screenshot(path=str(OUT / "00-lobby.png"))
            page.tap(f"[data-mode={args.mode}]")
            page.tap("#startbtn")
            t0, clicks, last_change, last_html, shots = time.time(), 0, time.time(), "", 0
            over = False
            while time.time() - t0 < args.max_s:
                if errors:
                    break
                if modal_visible(page):
                    over = True
                    break
                check_a_log_drawer(page, errors, ran)
                check_b_field_detail(page, errors, ran)
                check_c_sheet_info(page, errors, ran)
                check_d_closers(page, errors, ran)
                close_stray_overlays(page, errors)
                if errors:
                    break
                html = page.inner_html("body")
                if html != last_html:
                    last_html, last_change = html, time.time()
                elif time.time() - last_change > 30:
                    errors.append("stall: page unchanged for 30 s")
                    import signal as _sig
                    srv.send_signal(_sig.SIGUSR1)  # server prints all thread stacks to server.log
                    diag = page.evaluate("""() => { const s = document.querySelector('#sheet');
                        const o = document.querySelector('#sheet [data-opt]'); let top = null;
                        if (o) { const r = o.getBoundingClientRect();
                          const e = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
                          top = e && (e.id || e.className || e.tagName); }
                        return {sheet: s && s.className, app: document.querySelector('#app').className,
                                opts: document.querySelectorAll('[data-opt]').length, top,
                                layer: [...document.querySelectorAll('#layer > *')].map(e => e.id + ':' + e.innerHTML.length)}; }""")
                    errors.append(f"stall diag: {diag}")
                    try:
                        page.locator("#sheet [data-opt]").first.tap(timeout=2000)
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"stall tap: {str(e)[:600]}")
                    time.sleep(1)
                    break
                opts = [o for o in page.locator("[data-opt]:not([disabled])").all() if o.is_visible()]
                acts = [a for a in page.locator(".actionable").all() if a.is_visible()]
                target = None
                if opts and (not acts or rng.random() < 0.6):
                    target = rng.choice(opts)
                elif acts:
                    target = rng.choice(acts)
                if target is None:
                    page.wait_for_timeout(200)
                    continue
                if not tap(page, target):
                    continue  # went away, or a reveal took the tap: retry
                clicks += 1
                if clicks % args.shot_every == 0:
                    shots += 1
                    page.screenshot(path=str(OUT / f"{shots:02d}-play.png"))
                page.wait_for_timeout(120)
            page.screenshot(path=str(OUT / "99-end.png"))
            # Second game: in-game menu -> surrender -> game-over modal.
            surrender_ok = False
            lobby_ok = False
            if over and not errors:
                try:
                    page.locator("[data-act=newgame]").tap(timeout=5000)
                    page.wait_for_selector("#startbtn", timeout=5000)
                    if args.surrender_thinking:
                        page.tap("[data-mode=fast]")
                    page.tap("#startbtn")
                    if args.surrender_thinking:
                        t1 = time.time()
                        while time.time() - t1 < 60:
                            if page.locator(".pbar.bot .think").count():
                                break
                            page.wait_for_timeout(250)
                        surrender_ok = surrender_through_menu(
                            page, errors, 2, "surrender-while-thinking")
                    else:
                        page.wait_for_timeout(3000)
                        surrender_ok = surrender_through_menu(page, errors, 3, "surrender")
                    if surrender_ok:
                        page.screenshot(path=str(OUT / "98-surrendered.png"))
                        # Back to lobby from the game-over screen. The modal
                        # covers the bars, so the in-game menu button is
                        # unreachable behind it -- use the New game button.
                        try:
                            if modal_visible(page):
                                if not tap(page, page.locator("[data-act=newgame]"), timeout=5000):
                                    raise RuntimeError("newgame tap failed")
                            elif page.locator("#gamemenu").count() == 0:
                                if not tap(page, page.locator("[data-act=gamemenu]").first, timeout=5000):
                                    raise RuntimeError("gamemenu tap failed")
                                page.wait_for_timeout(300)
                                if not tap(page, page.locator("#gamemenu [data-act=backlobby]"), timeout=5000):
                                    raise RuntimeError("backlobby tap failed")
                            else:
                                if not tap(page, page.locator("#gamemenu [data-act=backlobby]"), timeout=5000):
                                    raise RuntimeError("backlobby tap failed")
                        except Exception as e:  # noqa: BLE001
                            errors.append(f"back-to-lobby failed: {e}")
                        page.wait_for_timeout(500)
                        lobby_ok = page.locator("#startbtn").count() > 0
                        if not lobby_ok:
                            errors.append("no lobby (#startbtn) after Back to lobby")
                except Exception as e:  # noqa: BLE001
                    errors.append(f"second-game flow failed: {e}")
            (OUT / "last-frames.json").write_text("\n".join(str(f) for f in frames[-3:]))
            browser.close()
        print(f"clicks={clicks} over={over} seconds={time.time() - t0:.0f} shots={shots}")
        print(f"checks ran: a-log-drawer={ran['a']} b-field-detail={ran['b']} c-sheet-info={ran['c']} d-closers={ran['d']}")
        print(f"surrender_ok={surrender_ok} lobby_ok={lobby_ok}")
        for e in errors[:10]:
            print("ERROR", e)
        need_checks = args.mode == "random"
        ok = over and not errors and surrender_ok and lobby_ok and (all(ran.values()) if need_checks else True)
        return 0 if ok else 2
    finally:
        srv.terminate()
        srv.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
