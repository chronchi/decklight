"""Local EDOPro-compatible host: a human plays our bot from EDOPro's LAN mode.

    uv run python scripts/edopro_host.py --deck decks/pool/snake-eye.ydk --port 7911 --opponent random
    uv run python scripts/edopro_host.py --deck decks/pool/mitsurugi-meta2.ydk --opponent product --once

Each user runs this on their own machine (default bind 127.0.0.1; --bind 0.0.0.0
to let someone on the LAN join). The duel runs in our engine; the bot holding the
other seat is --opponent random (a uniform random legal player) or product (the full
product search as scripts/search_match.py plays it: --fast = raw P, no search).
"""

from __future__ import annotations

import argparse
import random
import socket
import struct
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.decks import load_ydk
from ygo_zero.edopro.bot import RandomBot
from ygo_zero.edopro.host import DuelHost
from ygo_zero.engine.duel import Engine


def run_selftest(port: int, deck_codes: list[int], pool: set[str | int], seed: int = 0) -> dict:
    """Drive the human seat randomly from in-process (the manual smoke): lobby, RPS/TP,
    then random legal answers until STOC_DUEL_END. Returns win/retries."""
    from ygo_zero.edopro import protocol as P
    from ygo_zero.engine import constants as C
    from ygo_zero.engine import messages as M
    from ygo_zero.engine.actions import make_decision

    engine = Engine()
    rng = random.Random(seed)
    sock = socket.create_connection(("127.0.0.1", port))
    reader = P.FrameReader()
    win, retries, start_seen, duel_end = None, 0, False, False

    def send(proto: int, payload: bytes = b"") -> None:
        sock.sendall(P.pack_frame(proto, payload))

    def recv():
        data = sock.recv(65536)
        if not data:
            raise ConnectionError("server closed")
        return reader.feed(data)

    def decide(raw: bytes) -> bytes:
        dec = make_decision(M.parse_one(raw), engine.cards, lambda: pool)
        while True:
            opts = dec.options()
            resp = dec.choose(rng.randrange(len(opts)))
            if resp is not None:
                return resp

    send(P.CTOS_PLAYER_INFO, P.encode_name("selftest"))
    send(P.CTOS_JOIN_GAME, P.pack_join_game(P.PRO_VERSION, 0, P.client_version_bytes(*P.CLIENT_VERSION)))
    send(P.CTOS_UPDATE_DECK,
         struct.pack("<II", len(deck_codes), 0) + struct.pack(f"<{len(deck_codes)}I", *deck_codes))
    send(P.CTOS_HS_READY)
    while not duel_end:
        for proto, payload in recv():
            if proto == P.STOC_SELECT_HAND:
                send(P.CTOS_HAND_RESULT, bytes([rng.randint(1, 3)]))
            elif proto == P.STOC_SELECT_TP:
                send(P.CTOS_TP_RESULT, bytes([1]))
            elif proto == P.STOC_GAME_MSG:
                t = payload[0]
                if t == C.MSG_START:
                    start_seen = True
                elif t == C.MSG_WIN:
                    win = (payload[1], payload[2])
                elif t == C.MSG_RETRY:
                    retries += 1
                elif t in M.SELECT_TYPES:
                    send(P.CTOS_RESPONSE, decide(payload))
            elif proto == P.STOC_DUEL_END:
                duel_end = True
    sock.close()
    return {"win": win, "retries": retries, "start_seen": start_seen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", type=Path, required=True, help="bot's .ydk deck")
    ap.add_argument("--port", type=int, default=7911)
    ap.add_argument("--bind", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    ap.add_argument("--opponent", default="random", choices=["random", "product"])
    ap.add_argument("--ckpt", type=Path, default=None, help="product P checkpoint (default data/cache/P/best.pt)")
    ap.add_argument("--value", type=Path, default=None,
                    help="product value checkpoint (default data/cache/value_p2/best.pt)")
    ap.add_argument("--emb", type=Path, default=None, help="card embeddings (default qwen3-0.6b-d512)")
    ap.add_argument("--fast", action="store_true", help="product: raw P, no search")
    ap.add_argument("--device", default="cpu", help="product device (default cpu)")
    ap.add_argument("--replay-threads", type=int, default=1, help="thread-parallel world checks/branch replays")
    ap.add_argument("--seed", type=int, default=None, help="deal seed (random by default; printed)")
    ap.add_argument("--bot-second", action="store_true", help="the bot goes second when it wins RPS")
    ap.add_argument("--once", action="store_true", help="exit after the first completed duel")
    ap.add_argument("--selftest", action="store_true",
                    help="drive the human seat randomly from in-process (manual smoke); implies --once")
    ap.add_argument("--name", default="ygo-zero", help="room name shown in EDOPro's LAN list")
    a = ap.parse_args()

    bot_deck = load_ydk(a.deck)
    engine = Engine()
    missing = [c for c in (*bot_deck.main, *bot_deck.extra) if c not in engine.cards]
    if missing:
        sys.exit(f"bot deck has {len(missing)} codes missing from the card DB, e.g. {missing[:3]}")
    if a.opponent == "product":
        from ygo_zero.edopro.product_bot import ProductLauncher

        bot = ProductLauncher(ckpt=a.ckpt, value_ckpt=a.value, emb=a.emb, device=a.device,
                              fast=a.fast, replay_threads=a.replay_threads)
        print(f"product: ckpt {bot.ckpt} (sha256 {bot.ckpt_sha[:12]}), "
              f"value {bot.value_ckpt} (sha256 {(bot.value_sha or '')[:12]}), "
              f"fast={a.fast} device={a.device}", flush=True)
    else:
        bot = RandomBot()
    host = DuelHost(engine, bot_deck, bot, seed=a.seed, bot_second=a.bot_second, name=a.name)
    if a.selftest:
        if not a.once:
            print("--selftest implies --once", flush=True)
        mixed = list(bot_deck.main) + list(bot_deck.extra)
        client = threading.Thread(target=lambda: run_selftest(a.port, mixed, set(mixed)),
                                  daemon=True)
        # serve(once=True) exits after the first completed duel; the client drives it.
        srv = threading.Thread(target=host.serve, args=(a.bind, a.port, True), daemon=True)
        srv.start()
        import time

        time.sleep(0.5)  # let the listener come up
        client.start()
        srv.join(timeout=600)
        client.join(timeout=30)
    else:
        host.serve(a.bind, a.port, once=a.once)


if __name__ == "__main__":
    main()
