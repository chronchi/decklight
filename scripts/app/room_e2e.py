#!/usr/bin/env python3
"""A2b-3c room UI end to end: two app servers, two phone-sized browser pages, a room played by tapping.

    uv run --with playwright python scripts/app/room_e2e.py --rust-bin rust/target/release/decklight-serve \
        --out data/cache/app/claude/room_e2e

Page A (Kai) and page B (Rin) each talk to their own decklight-serve, like two phones. Steps (each must hold, else
exit 1):
  1. A: People tab, a nearby scan answers; Host -> the waiting room lists addresses, its seat, "Start" disabled.
  2. B: People tab; a malformed address gets the inline error; then B joins A by 127.0.0.1:<port>.
  3. Both see two seats; B taps Ready; A's Start enables; A starts.
  4. Rock-paper-scissors: A rock, B scissors -> A chooses "Go first"; both boards appear with the other's name.
  5. A's game menu shows Hint/Undo/What-if/Win chance "Off in rooms"; A surrenders: A "Defeat", B "Victory".
  6. Both tap "Back to the room": two seats again. A taps Close twice (confirm, guest present) -> People tab;
     B sees the room end.
Along the way: axe-core (WCAG 2.2 AA tags) on the People tab, both waiting rooms, RPS and game over; every room
button >= 44 px. Screenshots and a summary go to --out. Exits non-zero on any failed step or page error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "app"))
from room_gate import start_server  # noqa: E402
from web_a11y import TARGETS_JS, run_axe  # noqa: E402
from web_screens import launch_kwargs  # noqa: E402

PHONE = {"width": 412, "height": 915}


class Fail(Exception):
    pass


def check(cond, what: str) -> None:
    if not cond:
        raise Fail(what)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rust-bin", type=Path, default=ROOT / "rust/target/release/decklight-serve")
    ap.add_argument("--out", type=Path, default=ROOT / "data/cache/app/claude/room_e2e")
    ap.add_argument("--no-axe", action="store_true", help="skip axe-core (it is fetched from cdnjs)")
    args = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"steps": [], "axe": {}, "small_targets": [], "page_errors": []}
    procs = []
    frames: dict[str, list] = {"A": [], "B": []}  # every WebSocket frame each page received (trimmed), for failures
    t0 = time.time()

    def step(name: str) -> None:
        summary["steps"].append({"step": name, "t": round(time.time() - t0, 1)})
        print(f"[{time.time() - t0:6.1f}s] {name}", flush=True)

    try:
        pa, port_a, tok_a = start_server(args.rust_bin, out, "a")
        procs.append(pa)
        pb, port_b, tok_b = start_server(args.rust_bin, out, "b")
        procs.append(pb)
        with sync_playwright() as pw:
            br = pw.chromium.launch(**launch_kwargs())
            A = br.new_page(viewport=PHONE, device_scale_factor=2, has_touch=True)
            B = br.new_page(viewport=PHONE, device_scale_factor=2, has_touch=True)
            for tag, page in (("A", A), ("B", B)):
                page.on("pageerror", lambda e, tag=tag: summary["page_errors"].append(f"{tag}: {e}"))
                page.on("websocket", lambda ws, tag=tag: ws.on(
                    "framereceived", lambda f, tag=tag: frames[tag].append(str(f)[:60] + " … " + str(f)[-260:])))
            A.goto(f"http://127.0.0.1:{port_a}/?t={tok_a}")
            B.goto(f"http://127.0.0.1:{port_b}/?t={tok_b}")

            def shot(page, name: str) -> None:
                page.wait_for_timeout(250)
                page.screenshot(path=str(out / f"{name}.png"))

            def audit(page, name: str) -> None:
                small = [
                    t for t in page.evaluate(TARGETS_JS)
                    if ("rbtn" in t["cls"] or "nearby" in t["cls"] or t["cls"].startswith("rinput")) and min(t["w"], t["h"]) < 44
                ]
                summary["small_targets"] += [dict(t, screen=name) for t in small]
                if not args.no_axe:
                    v = run_axe(page)
                    summary["axe"][name] = v
                    for x in v:
                        print(f"  axe {name}: {x['id']} ({x['impact']}) {x['nodes'][:2]}")

            # 1. host
            for page in (A, B):
                page.wait_for_selector("nav.bbar")
                page.click('[data-act=lobbytab][data-tab=people]')
                page.wait_for_selector("#rname")
            A.fill("#rname", "Kai")
            B.fill("#rname", "Rin")
            A.wait_for_function("!document.querySelector('[data-act=roomscan][aria-busy]')", timeout=15000)
            step("people tab, scan answered")
            shot(A, "01_people")
            audit(A, "people")
            A.click("[data-act=roomhost]")
            A.wait_for_selector(".roomview .addr code, .roomview .seats", timeout=20000)
            A.wait_for_selector(".roomview .seats li")
            addrs = A.locator(".addr code").all_inner_texts()
            check(addrs, "host lists no address")
            port = int(addrs[0].rsplit(":", 1)[1])
            check(A.locator("#startbtn").is_disabled(), "Start enabled with no guest")
            check("Kai" in A.locator(".seats").inner_text(), "host seat missing the duelist name")
            step(f"hosting on port {port}, addresses {addrs}")
            shot(A, "02_host_waiting")
            audit(A, "host_waiting")

            # 2. join (a bad address first)
            B.fill("#raddr", "not an address!")
            B.click("[data-act=roomjoin]")
            B.wait_for_selector("#raddr-err")
            check(B.get_attribute("#raddr", "aria-invalid") == "true", "bad address not marked aria-invalid")
            step("bad address: " + B.inner_text("#raddr-err"))
            shot(B, "03_bad_address")
            B.fill("#raddr", f"127.0.0.1:{port}")
            B.click("[data-act=roomjoin]")
            B.wait_for_selector("[data-act=roomready]", timeout=20000)
            A.wait_for_function("document.querySelectorAll('.roomview .seats li:not(.open)').length === 2", timeout=20000)
            check("Rin" in A.locator(".seats").inner_text(), "host does not see the guest's name")
            check("Kai" in B.locator(".seats").inner_text(), "guest does not see the host's name")
            step("joined: " + B.inner_text(".roomview .status"))
            shot(B, "04_guest_joined")
            audit(B, "guest_joined")

            # 3. ready, start
            B.click("[data-act=roomready]")
            A.wait_for_function("!document.querySelector('#startbtn').disabled", timeout=10000)
            step("guest ready, Start enabled: " + A.inner_text(".roomview .status"))
            shot(A, "05_host_ready")
            A.click("#startbtn")

            # 4. rock-paper-scissors and turn order
            for page in (A, B):
                page.wait_for_selector("[data-act=roomrps]", timeout=30000)
            shot(A, "06_rps")
            audit(A, "rps")
            A.click("[data-act=roomrps][data-hand='2']")  # rock
            B.click("[data-act=roomrps][data-hand='1']")  # scissors
            A.wait_for_selector("[data-act=roomfirst]", timeout=15000)
            shot(A, "07_first")
            A.click("[data-act=roomfirst][data-tp='1']")
            for page in (A, B):
                page.wait_for_selector("[data-act=gamemenu]", timeout=60000)
            A.wait_for_timeout(1500)
            bar_a = A.inner_text("#app")
            bar_b = B.inner_text("#app")
            check("Rin" in bar_a, "A's board does not name Rin")
            check("Kai" in bar_b, "B's board does not name Kai")
            check("decklight" not in A.inner_text(".bar.bot") if A.locator(".bar.bot").count() else True,
                  "A's opponent bar still says decklight")
            step("both boards up")
            shot(A, "08_duel_A")
            shot(B, "09_duel_B")

            # 5. menu, surrender
            A.click("[data-act=gamemenu]")
            A.wait_for_selector("#gamemenu")
            menu = A.inner_text("#gamemenu")
            for word in ("Hint", "Undo", "What-if", "Win chance", "Off in rooms", "Leave room", "Surrender"):
                check(word in menu, f"room menu lacks {word!r}")
            check("Opponent's deck" not in menu and "Back to lobby" not in menu, "room menu shows bot-game rows")
            shot(A, "10_menu")
            A.click("#gamemenu [data-act=surrender]")
            A.click("#gamemenu [data-act=surrender]")
            for page in (A, B):
                page.wait_for_selector("#gameover [data-act=roomback]", timeout=30000)
            over_a, over_b = A.inner_text("#gameover"), B.inner_text("#gameover")
            check("Defeat" in over_a and "Victory" in over_b, f"results disagree: A={over_a!r} B={over_b!r}")
            check("Rin" in over_a and "Kai" in over_b, "game over lacks the opponent's name")
            step("surrender: A " + over_a.split("\n")[1] + ", B " + over_b.split("\n")[1])
            shot(A, "11_over_A")
            shot(B, "12_over_B")
            audit(B, "game_over")

            # 6. back to the room, then the host closes it
            for page in (A, B):
                page.click("#gameover [data-act=roomback]")
            A.wait_for_function("document.querySelectorAll('.roomview .seats li:not(.open)').length === 2", timeout=30000)
            B.wait_for_selector("[data-act=roomready]", timeout=30000)
            step("back in the room: " + A.inner_text(".roomview .status"))
            shot(A, "13_back_room")
            A.click(".rtop [data-act=roomleave]")
            A.wait_for_selector(".rtop .confirm")
            check("disconnected" in A.inner_text(".roomview"), "close asks no confirmation with a guest present")
            shot(A, "14_close_confirm")
            A.click(".rtop [data-act=roomleave]")
            A.wait_for_selector("#rname", timeout=10000)
            B.wait_for_function(
                "!document.querySelector('.roomview') || !!document.querySelector('.roomview .err') "
                "|| /over|closed|left/i.test(document.querySelector('.roomview').innerText)",
                timeout=30000,
            )
            step("host closed; guest: " + B.inner_text("#app")[:120].replace("\n", " | "))
            shot(B, "15_guest_after_close")
            br.close()
    except Fail as e:
        summary["failed"] = str(e)
    except Exception as e:  # noqa: BLE001
        summary["failed"] = f"{type(e).__name__}: {e}"
    finally:
        for p in procs:
            p.kill()
        for tag, fs in frames.items():
            (out / f"frames_{tag}.txt").write_text("\n".join(fs))
    summary["axe_violations"] = sum(len(v) for v in summary["axe"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    bad = summary.get("failed") or summary["page_errors"] or summary["small_targets"] or summary["axe_violations"]
    print(json.dumps({k: summary[k] for k in ("page_errors", "small_targets", "axe_violations")} | {"failed": summary.get("failed")}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
