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
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks import load_ydk
from decklight.edopro.bot import RandomBot
from decklight.edopro.host import DuelHost
from decklight.engine.duel import Engine


def run_selftest(port: int, deck_codes: list[int], pool: set[str | int], seed: int = 0) -> dict:
    """Drive the human seat with ClientSession + RandomAgent (manual smoke).

    Same output dict as before: win, retries, start_seen.
    """
    from decklight.edopro.client import ClientSession, DecisionOnlyView
    from decklight.play.agents import RandomAgent

    # +1: the host bot defaults to seed 0, and identical RPS sequences would
    # tie forever (both sides re-throw in lockstep).
    agent = RandomAgent(seed + 1)
    session = ClientSession(
        "127.0.0.1", port, "selftest", (list(deck_codes), [], []), agent,
        view_factory=lambda seat, codes: DecisionOnlyView(
            seat, codes, pool=set(pool) | set(codes)),
        timeout_s=600.0,
    )
    code = session.run()
    if code != 0:
        print(f"selftest client exited with code {code}", flush=True)
    return {"win": session.record.winner, "retries": session.record.retries,
            "start_seen": session.record.seat is not None}


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
        from decklight.edopro.product_bot import ProductLauncher

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
