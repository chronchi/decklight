#!/usr/bin/env python3
"""Record seats e2e (Ask my_stats needs it, NOTEBOOK 2026-11-14): against the real Rust server, start games with the
bot going first and with rock-paper-scissors until the human has sat in both seats, surrender each, and check that
- the primer names the bot's deck (not ours) whoever goes first;
- the record's deck names are in seat order (ours on the human seat);
- the index row has our seat (`me`), our deck and the opponent's, and `won` = 0 (we surrendered).

    uv run python scripts/app/record_seats_e2e.py [--rust-bin rust/target/release/decklight-serve]

Exit 1 on any failed check (ALERT line), 0 when every check passes. Smoke-sized (< 1 min on the laptop).
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_gate import start_server  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ME, BOT = ("blitzclique-meta1", "Blitzclique"), ("lunalight-meta1", "Lunalight")


def fail(msg: str) -> None:
    print(f"ALERT record_seats_e2e: {msg}", flush=True)
    raise SystemExit(1)


async def frame(ws, want: str, timeout=60.0) -> dict:
    while True:
        f = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if f.get("t") == "error":
            fail(f"server error while waiting for {want}: {f}")
        if f.get("t") == want:
            return f


async def one_game(ws, records: Path, first: str) -> int:
    before = set(records.glob("*.match.json.gz"))
    await ws.send(json.dumps({"t": "new_game", "deck": ME[0], "bot_deck": BOT[0], "first": first, "mode": "random"}))
    await frame(ws, "state")
    await ws.send(json.dumps({"t": "primer"}))
    p = await frame(ws, "primer")
    if p.get("deck") != BOT[1]:
        fail(f"first={first}: primer names {p.get('deck')!r}, not the bot's deck {BOT[1]!r}")
    await ws.send(json.dumps({"t": "surrender"}))
    for _ in range(100):
        new = set(records.glob("*.match.json.gz")) - before
        if new:
            break
        await asyncio.sleep(0.1)
    else:
        fail(f"first={first}: no record written after surrender")
    path = new.pop()
    rec = json.load(gzip.open(path))
    seat = next(i for i, s in enumerate(rec["seats"]) if s.get("type") == "human")
    names = [d.get("name") for d in rec["decks"]]
    if names[seat] != ME[1] or names[1 - seat] != BOT[1]:
        fail(f"first={first}: human seat {seat} but record deck names {names}")
    con = sqlite3.connect(records / "index.sqlite")
    row = con.execute("SELECT me, won, my_deck, opp_deck, opp_type FROM matches WHERE id = ?", (rec["id"],)).fetchone()
    if row != (seat, 0, ME[1], BOT[1], "bot"):
        fail(f"first={first}: index row {row}, want ({seat}, 0, {ME[1]!r}, {BOT[1]!r}, 'bot')")
    print(f"ok   first={first}: human seat {seat}, names {names}, index {row}", flush=True)
    return seat


async def run(port: int, token: str, records: Path) -> None:
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws?t={token}", max_size=None) as ws:
        seats = {await one_game(ws, records, "bot")}
        for i in range(12):
            if len(seats) == 2:
                break
            seats.add(await one_game(ws, records, "rps"))
        if seats != {0, 1}:
            fail(f"the human never sat in both seats: {seats}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust-bin", type=Path, default=ROOT / "rust/target/release/decklight-serve")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "data/cache/record_seats_e2e", help="server log and records")
    a = ap.parse_args()
    shutil.rmtree(a.run_dir / "records_seats", ignore_errors=True)
    proc, port, token = start_server(a.rust_bin, a.run_dir, "seats")
    try:
        asyncio.run(run(port, token, a.run_dir / "records_seats"))
    finally:
        proc.terminate()
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
