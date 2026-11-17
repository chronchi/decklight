#!/usr/bin/env python3
"""A2b-3c room gate: two app servers, one hosts a room and one joins it, both played by script over WebSocket.

    room_gate.py --rust-bin rust/target/release/decklight-serve --pair A:B --seed S --out DIR [--duels N]

Steps (each must hold, else exit 1):
  1. app A sends `room_host`; its room frame lists the port, at least one address and its own seat.
  2. a third page (app C) joins with a wrong password -> "Wrong password"; A's room survives (back to waiting).
  3. app B joins with the right password; both see two seats; B readies; A starts.
  4. both answer rock-paper-scissors / turn order, then play random legal options until the duel ends:
     results must agree (one "you", one "opponent"), both sides must have made decisions.
  5. `room_back` on both, a second duel, which B surrenders after a few decisions.
Writes every frame to DIR/frames.jsonl.gz and a summary to DIR/summary.json.

  --vs edopro-join: app A hosts, our standalone EDOPro client (`edopro-join`, StreamBot) joins and plays; one duel.
  --vs multirole:   a local Multirole (scripts/multirole_local.sh, already built) holds a room created by
                    `edopro-join --create`; app A joins it by `127.0.0.1:7911#<room id>` and plays; one duel.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class SplitMix:
    def __init__(self, seed: int) -> None:
        self.s = seed & (2**64 - 1)

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & (2**64 - 1)
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & (2**64 - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2**64 - 1)
        return z ^ (z >> 31)


def deck_text(name: str) -> str:
    p = Path(name)
    if not p.is_file():
        p = ROOT / "decks/pool" / f"{name}.ydk"
    return p.read_text()


def start_server(rust_bin: Path, run_dir: Path, tag: str) -> tuple[subprocess.Popen, int, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    log = (run_dir / f"server_{tag}.log").open("wb")
    cmd = [
        "taskset", "-c", "0-3", str(rust_bin), "--port", "0", "--token", token,
        "--records", str(run_dir / f"records_{tag}"), "--pics-dir", str(run_dir / "pics"), "--root", str(ROOT),
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=log)
    t0 = time.time()
    while time.time() - t0 < 60:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise SystemExit(f"server {tag} exited (rc={proc.returncode}); see {log.name}")
            time.sleep(0.05)
            continue
        text = line.decode(errors="replace").strip()
        if "listening on" in text:
            return proc, int(text.rsplit(" ", 1)[1]), token
    proc.kill()
    raise SystemExit(f"server {tag} did not start")


class Page:
    """One app page: a WebSocket plus the frames it received."""

    def __init__(self, tag: str, ws, sink) -> None:
        self.tag, self.ws, self.sink = tag, ws, sink
        self.room: dict = {}
        self.queue: list[dict] = []
        self.decisions = 0
        self.ticks = 0
        self.auto_passes = 0
        self.tick_secs: list[float] = []
        self.errors: list[str] = []

    async def send(self, msg: dict) -> None:
        self.sink({"page": self.tag, "sent": msg})
        await self.ws.send(json.dumps(msg))

    async def recv(self, timeout: float = 120.0) -> dict:
        frame = json.loads(await asyncio.wait_for(self.ws.recv(), timeout))
        self.sink({"page": self.tag, "frame": frame})
        if frame.get("t") == "room":
            self.room = frame
        if frame.get("t") == "error":
            self.errors.append(frame.get("text", ""))
        return frame

    async def until(self, pred, what: str, timeout: float = 60.0) -> dict:
        end = time.time() + timeout
        while time.time() < end:
            f = await self.recv(max(0.1, end - time.time()))
            if pred(f):
                return f
        raise SystemExit(f"{self.tag}: timed out waiting for {what}; last room={self.room}")


def fail(msg: str) -> None:
    print(f"ALERT room_gate: {msg}", file=sys.stderr)
    raise SystemExit(1)


def seats_taken(room: dict) -> int:
    return sum(1 for s in room.get("seats") or [] if s)


async def play(page: Page, rng: SplitMix, surrender_after: int | None, slow: int = 0) -> dict:
    """Answer asks and prompts until this page's duel has a result; returns the final state frame."""
    while True:
        f = await page.recv(300)
        t = f.get("t")
        if t == "room":
            ask = f.get("ask")
            if ask == "rps":
                await page.send({"t": "room_rps", "hand": 1 + rng.next() % 3})
            elif ask == "first":
                await page.send({"t": "room_first", "first": True})
            elif f.get("state") == "ended" and f.get("error"):
                fail(f"{page.tag}: room ended early: {f['error']}")
            continue
        if t == "error":
            fail(f"{page.tag}: server error {f}")
        if t != "state":
            continue
        if f.get("thinking") is not None:
            page.ticks += 1
            secs = f["thinking"].get("seconds", 0)
            if page.tick_secs and secs < page.tick_secs[-1] and secs > 1.5:
                pass  # a new wait restarts the count
            page.tick_secs.append(secs)
            if not f["thinking"].get("who"):
                fail(f"{page.tag}: thinking frame without who: {f['thinking']}")
            continue
        page.auto_passes += sum(1 for e in f.get("log") or [] if e.get("kind") == "auto_pass")
        if f.get("result") is not None:
            return f
        if f.get("prompt") is not None:
            if surrender_after is not None and page.decisions >= surrender_after:
                await page.send({"t": "surrender"})
                surrender_after = None
                continue
            n = len(f["prompt"]["options"])
            if page.decisions < slow:
                await asyncio.sleep(2.5)
            page.decisions += 1
            await page.send({"t": "choose", "prompt": f["prompt"]["id"], "index": rng.next() % n})


async def gate(args) -> dict:
    import websockets

    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    out = gzip.open(run / "frames.jsonl.gz", "wt")
    n_lines = 0

    def sink(rec: dict) -> None:
        nonlocal n_lines
        out.write(json.dumps(rec, separators=(",", ":")) + "\n")
        n_lines += 1

    a_name, b_name = args.pair.split(":")
    deck_a, deck_b = deck_text(a_name), deck_text(b_name)
    procs = []
    summary: dict = {"pair": args.pair, "seed": args.seed}
    t0 = time.time()
    try:
        pa, port_a, tok_a = start_server(args.rust_bin, run, "a")
        procs.append(pa)
        pb, port_b, tok_b = start_server(args.rust_bin, run, "b")
        procs.append(pb)
        url = "ws://127.0.0.1:{}/ws?t={}"
        async with websockets.connect(url.format(port_a, tok_a), max_size=None) as wa, websockets.connect(
            url.format(port_b, tok_b), max_size=None
        ) as wb:
            A, B = Page("A", wa, sink), Page("B", wb, sink)
            await A.recv()  # lobby
            await B.recv()

            # 1. host
            await A.send({"t": "room_host", "deck": {"ydk": deck_a}, "name": "Kai", "password": "pw1"})
            r = await A.until(lambda f: f.get("t") == "room" and any(s and s.get("you") for s in f.get("seats") or []), "host seat")
            server = r["server"]
            port = int(server.rsplit(":", 1)[1])
            if r["role"] != "host" or not r.get("addresses") or r["note"] != "Games may be shared anonymously for research":
                fail(f"host room frame incomplete: {r}")
            summary["room_port"] = port
            summary["addresses"] = r["addresses"]

            # LAN discovery finds the room
            await B.send({"t": "room_scan"})
            r = await B.until(lambda f: f.get("t") == "rooms", "scan result", 15)
            summary["scan"] = r["rooms"]
            if not any(x.get("name") == "Kai's room" and x["address"].endswith(f":{port}") for x in r["rooms"]):
                fail(f"LAN scan did not find Kai's room: {r}")

            # 2. wrong password from a second page of app B (then B's real page joins)
            await B.send({"t": "room_join", "address": f"127.0.0.1:{port}", "password": "nope", "deck": {"ydk": deck_b}, "name": "Ren"})
            r = await B.until(lambda f: f.get("t") == "room" and f.get("state") == "ended", "wrong-password end")
            if r.get("error") != "Wrong password":
                fail(f"wrong password gave {r.get('error')!r}")
            await B.send({"t": "room_leave"})
            await B.until(lambda f: f.get("t") == "lobby", "lobby after leave")
            # The host rejoins its own room after the failed lobby.
            await A.until(lambda f: f.get("t") == "room" and f.get("state") == "waiting" and seats_taken(f) == 1
                          and any(s and s.get("you") for s in f["seats"]), "host back to waiting", 20)
            await asyncio.sleep(0.5)

            # invalid address is a field error
            await B.send({"t": "room_join", "address": "a b", "deck": {"ydk": deck_b}})
            r = await B.until(lambda f: f.get("t") == "room", "address error")
            if r.get("field") != "address":
                fail(f"bad address gave {r}")

            for duel in range(args.duels):
                # 3. join, ready, start
                if duel == 0:
                    await B.send({"t": "room_join", "address": f"127.0.0.1:{port}", "password": "pw1", "deck": {"ydk": deck_b}, "name": "Ren"})
                await B.until(lambda f: f.get("t") == "room" and f.get("state") == "waiting" and seats_taken(f) == 2 and f.get("name") == "Kai's room", "guest sees two seats", 30)
                await A.until(lambda f: f.get("t") == "room" and f.get("state") == "waiting" and seats_taken(f) == 2, "host sees the guest", 30)
                await B.send({"t": "room_ready", "ready": True})
                await A.until(lambda f: f.get("t") == "room" and f.get("state") == "waiting" and any(s and not s["you"] and s["ready"] for s in f["seats"]), "guest ready", 30)
                await A.send({"t": "room_start"})

                # 4. duel
                rng_a, rng_b = SplitMix(args.seed ^ duel), SplitMix(args.seed ^ 0xA5A5 ^ duel)
                surrender = 30 if duel == 1 else None
                slow = 2 if duel == 1 else 0
                fa, fb = await asyncio.gather(play(A, rng_a, None, slow), play(B, rng_b, surrender))
                wa_, wb_ = fa["result"]["winner"], fb["result"]["winner"]
                rec = {"duel": duel, "a": fa["result"], "b": fb["result"], "a_decisions": A.decisions, "b_decisions": B.decisions,
                       "a_ticks": A.ticks, "b_ticks": B.ticks, "a_auto_passes": A.auto_passes, "b_auto_passes": B.auto_passes}
                summary.setdefault("duels", []).append(rec)
                print(json.dumps(rec), file=sys.stderr, flush=True)
                if {wa_, wb_} != {"you", "opponent"}:
                    fail(f"duel {duel}: results disagree {fa['result']} vs {fb['result']}")
                if A.decisions == 0 or B.decisions == 0:
                    fail(f"duel {duel}: a side made no decision")
                if duel == 1 and (B.ticks < 2 or max(B.tick_secs) < 2.0):
                    fail(f"duel {duel}: B saw {B.ticks} thinking frames {B.tick_secs} while A waited 2.5 s twice")
                if duel == 1 and fb["result"].get("reason") != "surrender":
                    fail(f"duel {duel}: B surrendered but the reason is {fb['result']}")
                if duel == 1 and wb_ != "opponent":
                    fail(f"duel {duel}: B surrendered but result is {fb['result']}")
                if fa["view"] is None or fb["view"] is None:
                    fail("state frame without a view")
                await A.until(lambda f: f.get("t") == "room" and f.get("state") == "ended", "host room ended", 30)
                await B.until(lambda f: f.get("t") == "room" and f.get("state") == "ended", "guest room ended", 30)
                rec["b_tick_secs"] = B.tick_secs[:20]
                A.decisions = B.decisions = A.ticks = B.ticks = A.auto_passes = B.auto_passes = 0
                A.tick_secs, B.tick_secs = [], []
                if duel + 1 < args.duels:
                    # 5. back to the room
                    await A.send({"t": "room_back"})
                    await A.until(lambda f: f.get("t") == "room" and f.get("state") == "waiting" and any(s and s.get("you") for s in f.get("seats") or []), "host back", 30)
                    await B.send({"t": "room_back"})
            for p in (A, B):
                # stale-prompt/"not your turn" errors are expected noise from racing clicks; nothing else is
                bad = [e for e in p.errors if e not in ("it is not your turn to choose",)]
                if bad:
                    fail(f"{p.tag} errors: {bad}")
            await A.send({"t": "room_leave"})
            await B.send({"t": "room_leave"})
    finally:
        out.close()
        for p in procs:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    summary["frames"] = n_lines
    summary["wall_s"] = round(time.time() - t0, 1)
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def edopro_join_cmd(args, deck: str, run: Path, extra: list[str]) -> list[str]:
    deck_path = Path(deck) if Path(deck).is_file() else ROOT / "decks/pool" / f"{deck}.ydk"
    return ["taskset", "-c", "0-3", str(args.rust_bin.parent / "edopro-join"), "--deck", str(deck_path),
            "--agent-seed", str(args.seed ^ 0x5BD1), "--record-dir", str(run / "client"), "--timeout", "60",  # not the app script's stream: equal throws tie forever
            "--packet-log", str(run / "client_packets.jsonl.gz"), *extra]


async def gate_vs_client(args) -> dict:
    """App A against a non-app EDOPro peer: `edopro-join` joining A's room, or A joining a Multirole room."""
    import urllib.request

    import websockets

    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    out = gzip.open(run / "frames.jsonl.gz", "wt")
    n_lines = 0

    def sink(rec: dict) -> None:
        nonlocal n_lines
        out.write(json.dumps(rec, separators=(",", ":")) + "\n")
        n_lines += 1

    shutil.rmtree(run / "client", ignore_errors=True)  # the result check reads the newest client record
    a_name, b_name = args.pair.split(":")
    procs: list[subprocess.Popen] = []
    summary: dict = {"pair": args.pair, "seed": args.seed, "vs": args.vs}
    t0 = time.time()
    client = None
    try:
        pa, port_a, tok_a = start_server(args.rust_bin, run, "a")
        procs.append(pa)
        async with websockets.connect(f"ws://127.0.0.1:{port_a}/ws?t={tok_a}", max_size=None) as wa:
            A = Page("A", wa, sink)
            await A.recv()
            clog = (run / "client.log").open("wb")
            if args.vs == "edopro-join":
                await A.send({"t": "room_host", "deck": {"ydk": deck_text(a_name)}, "name": "Kai", "password": "pw2"})
                r = await A.until(lambda f: f.get("t") == "room" and f.get("state") == "waiting", "room open")
                port = int(r["server"].rsplit(":", 1)[1])
                client = subprocess.Popen(edopro_join_cmd(args, b_name, run, ["--host", "127.0.0.1", "--port", str(port),
                                          "--password", "pw2", "--name", "Ren"]), cwd=str(ROOT), stdout=clog, stderr=clog)
                await A.until(lambda f: f.get("t") == "room" and f.get("state") == "waiting" and
                              any(s and not s["you"] and s["ready"] for s in f.get("seats") or []), "edopro-join ready", 60)
                await A.send({"t": "room_start"})
            else:
                client = subprocess.Popen(edopro_join_cmd(args, b_name, run, ["--host", "127.0.0.1", "--port", "7911",
                                          "--create", "--name", "Ren", "--password", "pw3"]),
                                          cwd=str(ROOT), stdout=clog, stderr=clog)
                room_id = None
                for _ in range(100):
                    await asyncio.sleep(0.2)
                    try:
                        listing = json.loads(urllib.request.urlopen("http://127.0.0.1:7922/api/getrooms", timeout=2).read())
                    except OSError:
                        continue
                    rooms = listing.get("rooms") or []
                    if rooms:
                        room_id = rooms[0]["roomid"]
                        summary["listing"] = rooms[0]
                        break
                if room_id is None:
                    fail("Multirole lists no room created by edopro-join")
                await A.send({"t": "room_join", "address": f"127.0.0.1:7911#{room_id}", "password": "pw3",
                              "deck": {"ydk": deck_text(a_name)}, "name": "Kai"})
                await A.until(lambda f: f.get("t") == "room" and f.get("state") == "waiting" and seats_taken(f) == 2, "two seats", 60)
                await A.send({"t": "room_ready", "ready": True})
            fa = await play(A, SplitMix(args.seed), None)
            summary["result"] = fa["result"]
            summary["decisions"] = A.decisions
            summary["auto_passes"] = A.auto_passes
            summary["ticks"] = A.ticks
            print(json.dumps({k: summary[k] for k in ("result", "decisions", "auto_passes", "ticks")}), file=sys.stderr)
            if A.decisions == 0:
                fail("the app made no decision")
            rc = client.wait(timeout=120)
            recs = sorted((run / "client").glob("*.json"))
            crec = json.loads(recs[-1].read_text()) if recs else {}
            summary["client_rc"] = rc
            summary["client_winner"] = crec.get("winner")
            summary["client_retries"] = crec.get("retries")
            if rc != 0 or crec.get("retries"):
                fail(f"edopro-join rc={rc} retries={crec.get('retries')} alerts={crec.get('alerts')}")
            cw = crec.get("winner")
            cseat = crec.get("seat")
            if cw is None or cseat is None:
                fail(f"edopro-join record has no winner/seat: {crec}")
            client_won = cw[0] == cseat
            if client_won != (fa["result"]["winner"] == "opponent"):
                fail(f"results disagree: app {fa['result']} vs client winner {cw} seat {cseat}")
            bad = [e for e in A.errors if e != "it is not your turn to choose"]
            if bad:
                fail(f"app errors: {bad}")
            await A.send({"t": "room_leave"})
    finally:
        out.close()
        if client is not None and client.poll() is None:
            client.kill()
        for p in procs:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    summary["frames"] = n_lines
    summary["wall_s"] = round(time.time() - t0, 1)
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust-bin", type=Path, default=ROOT / "rust/target/release/decklight-serve")
    ap.add_argument("--pair", default="blackwing:burning-abyss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duels", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vs", choices=("app", "edopro-join", "multirole"), default="app")
    args = ap.parse_args(argv)
    s = asyncio.run(gate(args) if args.vs == "app" else gate_vs_client(args))
    print(json.dumps({k: s[k] for k in ("pair", "seed", "frames", "wall_s")}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
