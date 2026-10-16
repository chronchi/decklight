"""Local EDOPro-compatible host: a human plays our bot from EDOPro's LAN mode.

    uv run python scripts/edopro_host.py --deck decks/pool/snake-eye.ydk --port 7911 --opponent random

Each user runs this on their own machine (default bind 127.0.0.1; --bind 0.0.0.0
to let someone on the LAN join). The duel runs in our engine; the bot holding the
other seat is --opponent random (a uniform random legal player; D3 plugs in the
real product behind the same Bot interface).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.decks import load_ydk
from ygo_zero.edopro.bot import RandomBot
from ygo_zero.edopro.host import DuelHost
from ygo_zero.engine.duel import Engine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", type=Path, required=True, help="bot's .ydk deck")
    ap.add_argument("--port", type=int, default=7911)
    ap.add_argument("--bind", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    ap.add_argument("--opponent", default="random", choices=["random"])
    ap.add_argument("--seed", type=int, default=None, help="deal seed (random by default; printed)")
    ap.add_argument("--bot-second", action="store_true", help="the bot goes second when it wins RPS")
    ap.add_argument("--once", action="store_true", help="exit after the first completed duel")
    ap.add_argument("--name", default="ygo-zero", help="room name shown in EDOPro's LAN list")
    a = ap.parse_args()

    bot_deck = load_ydk(a.deck)
    engine = Engine()
    missing = [c for c in (*bot_deck.main, *bot_deck.extra) if c not in engine.cards]
    if missing:
        sys.exit(f"bot deck has {len(missing)} codes missing from the card DB, e.g. {missing[:3]}")
    bot = RandomBot()
    host = DuelHost(engine, bot_deck, bot, seed=a.seed, bot_second=a.bot_second, name=a.name)
    host.serve(a.bind, a.port, once=a.once)


if __name__ == "__main__":
    main()
