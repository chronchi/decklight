"""End-to-end check against a real Multirole server: a room host and a joining bot play full duels.

Multirole is the server EDOPro's public lobbies run. On it a room is created by a player
(CTOS_CREATE_GAME) and a bot joins it by room id, which it reads from the room listing
(`http://<host>:<lobby-port>/`, JSON). This script plays that whole flow N times:

  1. host: our `ClientSession(create=HostInfo(...))` with the random agent creates a room tagged
     with a unique note and starts the duel when the joiner is ready;
  2. joiner: `scripts/edopro_join.py Deck=<bundle> Host= Port= RoomId= HostInfo=` (the
     WindBot-style command a user runs), room id taken from the listing;
  3. both must finish with exit 0, no MSG_RETRY, and agree on the winner.

    uv run python scripts/audit/multirole_check.py --games 3 --bot random-snake-eye
    uv run python scripts/audit/multirole_check.py --games 2 --bot raw-blueeyes
    # custom cards: start the server with the card repo (scripts/multirole_local.sh WORKDIR CARD_DIR),
    # both bundles list the same dir in `cards`
    uv run python scripts/audit/multirole_check.py --rule 4 --bot wiccanthrope-raw --host-bot wiccanthrope-random

Exits 1 when any duel fails one of the checks (fail loud: a retry means our view of the duel
disagrees with the server's).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decklight.edopro import protocol as P
from decklight.edopro.client import ClientSession
from decklight.play.bundle import discover_bundles
from decklight.play.stream import StreamView


def find_room(host: str, lobby_port: int, note: str, timeout_s: float) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{host}:{lobby_port}/", timeout=5) as r:
                rooms = json.load(r).get("rooms", [])
        except OSError:
            rooms = []
        for room in rooms:
            if room.get("roomnotes") == note:
                return int(room["roomid"])
        time.sleep(0.2)
    raise TimeoutError(f"room with notes {note!r} never appeared in the listing")


def custom_codes(bundle) -> set[int]:
    """Codes defined in the bundle's custom card dirs (empty for a bundle without `cards`)."""
    out: set[int] = set()
    for d in bundle.cards:
        for cdb in sorted(Path(d).glob("*.cdb")):
            with sqlite3.connect(f"file:{cdb}?mode=ro", uri=True) as con:
                out.update(r[0] for r in con.execute("SELECT id FROM datas"))
    return out


def last_record(d: Path) -> dict:
    files = sorted(d.glob("*.json"))
    if not files:
        return {}
    return json.loads(files[-1].read_text())


def one_duel(args, n: int, host_bundle, out: Path, custom: set[int]) -> dict:
    note = f"decklight-check-{uuid.uuid4().hex[:8]}"
    rec_host, rec_join = out / f"g{n}-host", out / f"g{n}-join"
    deck = host_bundle.load_deck()
    cards = host_bundle.engine().cards
    session = ClientSession(
        args.host, args.port, "check-host", deck, host_bundle.make_agent(seed=1000 + n),
        view_factory=lambda seat, codes: StreamView(
            {"main": list(deck.main), "extra": list(deck.extra)}, cards, seat=seat),
        password=args.password, timeout_s=args.timeout, record_dir=rec_host,
        create=P.HostInfo(rule=args.rule, start_lp=8000), room_notes=note,
    )
    result: dict = {}
    t = threading.Thread(target=lambda: result.__setitem__("host_exit", session.run()), daemon=True)
    t0 = time.monotonic()
    t.start()
    room = find_room(args.host, args.lobby_port, note, 20)
    cmd = [sys.executable, str(ROOT / "scripts/edopro_join.py"), "Name=decklight", f"Deck={args.bot}",
           f"Host={args.host}", f"Port={args.port}", f"RoomId={room}", f"HostInfo={args.password}",
           "--record-dir", str(rec_join), "--timeout", str(args.timeout)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.max_s, check=False)
    t.join(args.max_s)
    hr, jr = last_record(rec_host), last_record(rec_join)
    result.update({
        "game": n, "room": room, "join_exit": p.returncode, "seconds": round(time.monotonic() - t0, 1),
        "host_winner": hr.get("winner"), "join_winner": jr.get("winner"),
        "host_retries": hr.get("retries"), "join_retries": jr.get("retries"),
        "join_seat": jr.get("seat"), "join_responses": jr.get("n_responses", 0),
        "alerts": (hr.get("alerts") or []) + (jr.get("alerts") or []),
        # custom-card activations / summons seen by the joiner (both players): proof the server ran their scripts
        "custom_events": sum(v for k, v in (jr.get("events") or {}).items() if int(k.split(":")[1]) in custom),
        "join_tail": p.stdout[-400:] + p.stderr[-400:],
    })
    problems = []
    if result.get("host_exit") != 0 or p.returncode != 0:
        problems.append(f"exit codes host={result.get('host_exit')} join={p.returncode}")
    if result["host_retries"] or result["join_retries"]:
        problems.append(f"retries host={result['host_retries']} join={result['join_retries']}")
    if result["host_winner"] is None or result["host_winner"] != result["join_winner"]:
        problems.append(f"winners differ: host={result['host_winner']} join={result['join_winner']}")
    if not result["join_responses"]:
        problems.append("the joining bot made no decision")
    result["problems"] = problems
    return result


def custom_check(results: list[dict], custom: set[int]) -> str | None:
    """A custom-card run must show custom cards in play at least once over all duels."""
    if custom and not sum(r["custom_events"] for r in results):
        return "no custom card was activated or summoned in any duel"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7911, help="Multirole roomHostingPort")
    ap.add_argument("--lobby-port", type=int, default=7922, help="Multirole lobbyListingPort")
    ap.add_argument("--bot", default="random-snake-eye", help="bundle the joiner plays (edopro_join Deck=)")
    ap.add_argument("--host-bot", default="random-snake-eye", help="bundle the room host plays")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--password", default="")
    ap.add_argument("--rule", type=int, default=3,
                    help="allowed cards: 3 official + prerelease, 4 any (custom-card servers)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-s", type=float, default=900.0, help="cap per duel")
    ap.add_argument("--out", type=Path, default=ROOT / "data/cache/multirole_check")
    args = ap.parse_args()

    bundles = discover_bundles()
    if args.host_bot not in bundles:
        print(f"unknown host bundle {args.host_bot!r}; known: {sorted(bundles)}", file=sys.stderr)
        return 2
    out = args.out / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    join_bundle = bundles.get(args.bot)
    custom = custom_codes(join_bundle) if join_bundle else set()
    failed, results = 0, []
    for n in range(args.games):
        r = one_duel(args, n, bundles[args.host_bot], out, custom)
        results.append(r)
        failed += bool(r["problems"])
        print(json.dumps({k: v for k, v in r.items() if k != "join_tail" or r["problems"]}), flush=True)
    print(f"multirole check: {args.games - failed}/{args.games} duels ok (records in {out})", flush=True)
    bad = custom_check(results, custom)
    if bad:
        print(f"multirole check: {bad} ({len(custom)} custom codes in {args.bot})", file=sys.stderr)
    return 1 if failed or bad else 0


if __name__ == "__main__":
    sys.exit(main())
