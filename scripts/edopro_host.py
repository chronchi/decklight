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


def run_selftest(port: int, deck, engine, seed: int = 0) -> dict:
    """Drive the human seat with ClientSession + RandomAgent over StreamView (manual smoke).

    Same output dict as before: win, retries, start_seen.
    """
    from decklight.edopro.client import ClientSession
    from decklight.play.agents import RandomAgent
    from decklight.play.stream import StreamView

    # +1: the host bot defaults to seed 0, and identical RPS sequences would
    # tie forever (both sides re-throw in lockstep).
    agent = RandomAgent(seed + 1)
    session = ClientSession(
        "127.0.0.1", port, "selftest", deck, agent,
        view_factory=lambda seat, codes: StreamView(
            {"main": list(deck.main), "extra": list(deck.extra)}, engine.cards, seat=seat),
        timeout_s=600.0,
    )
    code = session.run()
    if code != 0:
        print(f"selftest client exited with code {code}", flush=True)
    return {"win": session.record.winner, "retries": session.record.retries,
            "start_seen": session.record.seat is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", type=Path, default=None, help="bot's .ydk deck (or use --bot)")
    ap.add_argument("--bot", default=None, help="bot bundle name or path (deck, agent, checkpoints, engine)")
    ap.add_argument("--bots-dir", action="append", default=[],
                    help="extra bundle dir (repeatable; default: bots/ + config/bots/)")
    ap.add_argument("--port", type=int, default=7911)
    ap.add_argument("--bind", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    ap.add_argument("--opponent", default="random", choices=["random", "product"])
    ap.add_argument("--ckpt", type=Path, default=None, help="product policy checkpoint (default data/cache/forge_r1/best.pt)")
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

    launcher = None
    emb_codes = None
    if a.bot:
        from decklight.play.bundle import BundleError, discover_bundles

        dirs = [Path(d) for d in a.bots_dir] + \
            [d for d in (ROOT / "bots", ROOT / "config" / "bots") if d.is_dir()]
        try:
            catalog = discover_bundles(dirs)
        except BundleError as e:
            sys.exit(str(e))
        from decklight.edopro import bots as B

        _path, bundle = B.resolve_deck(a.bot, catalog)
        if bundle is None:
            sys.exit(f"unknown bot {a.bot!r} (--list-bots in edopro_join.py lists the bundles)")
        bot_deck = bundle.load_deck()
        engine = bundle.engine()
        missing = [c for c in (*bot_deck.main, *bot_deck.extra) if c not in engine.cards]
        if missing:
            sys.exit(f"bot deck has {len(missing)} codes missing from the card DB, e.g. {missing[:3]}")
        import numpy as np

        from decklight.search.product import resolve_product_paths

        _, _, emb_p = resolve_product_paths(None, None, bundle._abs(bundle.emb))
        emb_codes = {int(c) for c in np.load(emb_p)["codes"]}
        if bundle.agent == "product":
            launcher = bundle.make_launcher()
            bot = None
            print(f"bundle {bundle.name}: product search (ckpt {launcher.ckpt})", flush=True)
        else:
            bot = bundle.make_bot()
            print(f"bundle {bundle.name}: agent {bundle.agent} on {bundle.deck_path()}", flush=True)
    else:
        if a.deck is None:
            sys.exit("need --deck or --bot")
        bot_deck = load_ydk(a.deck)
        engine = Engine()
        missing = [c for c in (*bot_deck.main, *bot_deck.extra) if c not in engine.cards]
        if missing:
            sys.exit(f"bot deck has {len(missing)} codes missing from the card DB, e.g. {missing[:3]}")
        if a.opponent == "product":
            from decklight.edopro.product_bot import ProductLauncher

            launcher = ProductLauncher(ckpt=a.ckpt, value_ckpt=a.value, emb=a.emb, device=a.device,
                                       fast=a.fast, replay_threads=a.replay_threads)
            bot = None
            print(f"product: ckpt {launcher.ckpt} (sha256 {launcher.ckpt_sha[:12]}), "
                  f"value {launcher.value_ckpt} (sha256 {(launcher.value_sha or '')[:12]}), "
                  f"fast={a.fast} device={a.device}", flush=True)
        else:
            bot = RandomBot()
    # A product launcher rides DuelHost.bot (Session branches on start_duel).
    host = DuelHost(engine, bot_deck, launcher if launcher is not None else bot, seed=a.seed,
                    bot_second=a.bot_second, name=a.name,
                    **({"emb_codes": emb_codes} if emb_codes is not None else {}))
    if a.selftest:
        if not a.once:
            print("--selftest implies --once", flush=True)
        client = threading.Thread(target=lambda: run_selftest(a.port, bot_deck, engine),
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
