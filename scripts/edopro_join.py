"""WindBot-compatible EDOPro joiner: play one seat of a room as a bot.

Single duel (WindBot `Key=Value` args and `--flags` both work)::

    uv run python scripts/edopro_join.py Name=test Deck=random-snake-eye Host=127.0.0.1 Port=7922
    uv run python scripts/edopro_join.py --deck decks/pool/snake-eye.ydk --host 127.0.0.1 --port 7922

`Deck=` is a bot bundle name (`bots/<name>/bot.toml`, `config/bots/<name>.toml`),
a bundle path, or a `.ydk` path (a default random bundle is built on the fly).
`Version=` is WindBot's packed int (EDOPro major | minor<<8 |
core major<<16 | core minor<<24); `--pro-version` / `--client-version`
("41.0.11.0") override it. `--agent/--ckpt/--emb/--device/--seed` override the
bundle's files.

Server mode (WindBot `ServerMode`)::

    uv run python scripts/edopro_join.py --serve 7923 --max-duels 1
    curl 'http://127.0.0.1:7923/?name=t&deck=random-snake-eye&host=127.0.0.1&port=7924'

Each request re-discovers the bundles (a bundle dropped in `bots/` works
without restart) and spawns one `edopro_join.py` subprocess (inheriting the
agent/ckpt/emb/device/record-dir/timeout flags) and returns HTTP 200 at once
(503 beyond `--max-duels`).

Every agent plays through `StreamView` (the raw policy over the network); the
product search needs a local engine, so `Deck=<product bundle>` exits 2 here.

Exit codes: 0 duel finished; 2 bad args/catalog or server-side rejection;
3 our view disagreed with the server (MSG_RETRY); 4 timeout/disconnect.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.edopro import bots as B
from decklight.edopro import protocol as P
from decklight.edopro.client import ClientSession
from decklight.play.bundle import BundleError, discover_bundles
from decklight.play.stream import StreamView

EXIT_OK, EXIT_ARGS, EXIT_RETRY, EXIT_TIMEOUT = 0, 2, 3, 4


def build_parser(kv: dict) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=kv.get("name", "decklight"))
    ap.add_argument("--deck", default=kv.get("deck"))
    ap.add_argument("--bot", default=None, help="bundle name or path (like Deck=)")
    ap.add_argument("--deckfile", default=kv.get("deckfile"))
    ap.add_argument("--bots-dir", action="append", default=[],
                    help="extra bundle dir (repeatable; default: bots/ + config/bots/)")
    ap.add_argument("--host", default=kv.get("host", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=kv.get("port", 7911))
    ap.add_argument("--password", default=kv.get("password", kv.get("hostinfo", "")),
                    help="room password (WindBot HostInfo=, sent in JOIN_GAME like WindBot's GameClient)")
    ap.add_argument("--roomid", type=int, default=kv.get("roomid", 0))
    ap.add_argument("--hand", type=int, default=kv.get("hand", 0),
                    help="forced RPS throw 1/2/3 (WindBot Hand=); 0 = agent decides")
    ap.add_argument("--version", type=int, default=kv.get("version"),
                    help="WindBot packed Version int (maps to ProtocolConfig)")
    ap.add_argument("--pro-version", default=None, help="u16 proto version, e.g. 0x1354")
    ap.add_argument("--client-version", default=None, help="dotted client version, e.g. 41.0.11.0")
    ap.add_argument("--agent", default=None, help="override the bundle's agent (random|raw)")
    ap.add_argument("--seed", type=int, default=None,
                    help="agent RNG seed (default: the bundle's, else random; equal seeds on both sides tie RPS forever)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--emb", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--record-dir", default="data/cache/edopro_client")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--debug", action="store_true", default=bool(kv.get("debug", False)))
    ap.add_argument("--serve", type=int, default=None, metavar="PORT")
    ap.add_argument("--max-duels", type=int, default=8)
    ap.add_argument("--list-bots", action="store_true")
    return ap


def make_proto(args) -> P.ProtocolConfig:
    if args.version is not None:
        try:
            proto = P.ProtocolConfig.from_windbot_version(args.version)
        except ValueError as e:
            sys.exit(f"bad Version=: {e}")
    else:
        proto = P.ProtocolConfig()
    if args.pro_version is not None:
        try:
            pro = int(args.pro_version, 0)
        except ValueError:
            sys.exit(f"bad --pro-version {args.pro_version!r} (want 0x1354 or 4948)")
        proto = P.ProtocolConfig(pro_version=pro, client_version=proto.client_version)
    if args.client_version is not None:
        try:
            client = P.ProtocolConfig.parse_version(args.client_version)
        except ValueError as e:
            sys.exit(f"bad --client-version: {e}")
        proto = P.ProtocolConfig(pro_version=proto.pro_version, client_version=client)
    return proto


def bundle_dirs(args) -> list[Path]:
    dirs = [Path(d) for d in (args.bots_dir or [])]
    for d in (ROOT / "bots", ROOT / "config" / "bots"):
        if d.is_dir():
            dirs.append(d)
    return dirs


def run_duel(args, catalog: dict | None = None) -> int:
    from decklight.play.agents import MissingCapability

    try:
        catalog = catalog if catalog is not None else discover_bundles(bundle_dirs(args))
    except BundleError as e:
        print(f"{e}", file=sys.stderr)
        return EXIT_ARGS
    ref = args.bot or args.deck or args.deckfile
    if not ref:
        print("need a bot: Deck=<bundle name> or a .ydk path (--list-bots lists the bundles)",
              file=sys.stderr)
        return EXIT_ARGS
    try:
        _deck_path, bundle = B.resolve_deck(str(ref), catalog)
    except BundleError as e:
        print(f"{e}", file=sys.stderr)
        return EXIT_ARGS
    if bundle is None:
        print(f"unknown deck {ref!r} (not a bundle, no such .ydk file)", file=sys.stderr)
        return EXIT_ARGS
    agent_name = args.agent or bundle.agent
    if agent_name == "product":
        print("agent 'product' needs a local engine (search); "
              "the network client plays raw or random", file=sys.stderr)
        return EXIT_ARGS
    seed = args.seed
    if seed is None:
        seed = bundle.seed or random.randrange(2**31)
    cfg: dict = {"seed": seed}
    if args.ckpt:
        cfg["ckpt"] = args.ckpt
    if args.emb:
        cfg["emb"] = args.emb
    if args.device:
        cfg["device"] = args.device
    try:
        agent = replace(bundle, agent=agent_name).make_agent(**cfg)
    except BundleError as e:
        print(f"{e}", file=sys.stderr)
        return EXIT_ARGS
    except ValueError as e:
        print(f"cannot make agent {agent_name!r}: {e}", file=sys.stderr)
        return EXIT_ARGS
    try:
        deck = bundle.load_deck()
    except (OSError, ValueError) as e:
        print(f"cannot load deck {bundle.deck_path()}: {e}", file=sys.stderr)
        return EXIT_ARGS
    engine = bundle.engine()
    missing = [c for c in (*deck.main, *deck.extra) if c not in engine.cards]
    if missing:
        print(f"deck has {len(missing)} codes missing from the card DB, e.g. {missing[:3]}",
              file=sys.stderr)
        return EXIT_ARGS
    proto = make_proto(args)
    if args.debug:
        print(f"join {args.host}:{args.port} as {args.name} bot={bundle.name} "
              f"agent={agent_name} proto={proto}", flush=True)
    session = ClientSession(
        args.host, args.port, args.name, deck, agent,
        view_factory=lambda seat, codes: StreamView(
            {"main": list(deck.main), "extra": list(deck.extra)}, engine.cards, seat=seat),
        proto=proto, password=args.password or "", gameid=args.roomid,
        hand=args.hand or 0, timeout_s=args.timeout, record_dir=args.record_dir,
    )
    try:
        return session.run()
    except MissingCapability as e:
        print(f"agent cannot run as a network client: {e}", file=sys.stderr)
        return EXIT_ARGS
    except NotImplementedError as e:
        print(f"agent needs more view than the client has: {e}", file=sys.stderr)
        return EXIT_ARGS


# --- --serve mode (WindBot ServerMode) ---

class ServeState:
    def __init__(self, args, script: Path) -> None:
        self.args = args
        self.script = script
        self.lock = threading.Lock()
        self.procs: list[subprocess.Popen] = []

    def reap(self) -> None:
        with self.lock:
            alive = []
            for p in self.procs:
                if p.poll() is None:
                    alive.append(p)
                else:
                    print(f"serve: duel exited with code {p.returncode}", flush=True)
            self.procs = alive

    def spawn(self, q: dict) -> bool:
        self.reap()
        with self.lock:
            if len(self.procs) >= self.args.max_duels:
                return False
            # Re-discover on each request: a bundle dropped in bots/ works without restart
            # (a broken bundle errors in the child, not here).
            argv = [sys.executable, str(self.script)]
            for key, flag in (("name", "Name"), ("deck", "Deck"), ("host", "Host"),
                              ("port", "Port"), ("password", "Password"),
                              ("version", "Version"), ("hand", "Hand")):
                if q.get(key) not in (None, ""):
                    argv.append(f"{flag}={q[key]}")
            for flag in ("agent", "ckpt", "emb", "device", "record-dir", "timeout",
                         "pro-version", "client-version"):
                val = getattr(self.args, flag.replace("-", "_"))
                if val not in (None, ""):
                    argv += [f"--{flag}", str(val)]
            for d in self.args.bots_dir:
                argv += ["--bots-dir", str(d)]
            if self.args.debug:
                argv.append("--debug")
            p = subprocess.Popen(argv)
            self.procs.append(p)
            print(f"serve: spawned {' '.join(argv)} (pid {p.pid})", flush=True)
            return True


def serve_forever(args) -> int:
    state = ServeState(args, Path(__file__).resolve())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = urllib.parse.urlsplit(self.path).query
            try:
                q = B.parse_serve_query(qs)
            except ValueError as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"bad argument: {e}\n".encode())
                return
            # WindBot requires name/host/port; we also require a deck.
            if not q.get("name") or not q.get("host") or q.get("port") is None \
                    or not (q.get("deck") or q.get("deckfile")):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"need ?name=&deck=&host=&port=\n")
                return
            if state.spawn(q):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok\n")
            else:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"too many duels\n")

        def log_message(self, *a):  # keep stdout for spawn/exit lines only
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", args.serve), Handler)
    print(f"serve: listening on 127.0.0.1:{args.serve} (max {args.max_duels} duels)", flush=True)

    def watcher():
        while True:
            time.sleep(5)
            state.reap()

    threading.Thread(target=watcher, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        kv, rest = B.parse_kv_args(argv)
    except ValueError as e:
        print(f"bad argument: {e}", file=sys.stderr)
        return EXIT_ARGS
    args = build_parser(kv).parse_args(rest)
    if args.list_bots:
        try:
            sys.stdout.write(B.list_bots_text(discover_bundles(bundle_dirs(args))))
        except BundleError as e:
            print(f"{e}", file=sys.stderr)
            return EXIT_ARGS
        return EXIT_OK
    if args.serve is not None:
        return serve_forever(args)
    return run_duel(args)


if __name__ == "__main__":
    sys.exit(main())
