"""Python reference side of the A2b-2 network gates, with packet logs.

Host mode wraps `DuelHost`/`Session` (the oracle host), join mode wraps `ClientSession`/`DecisionOnlyView`.
A StreamBot (splitmix64) plays the bot seat / the client, matching `rust/decklight-edopro`'s `StreamBot`, so a
Python and a Rust process make the same choices for the same seed.

    uv run python scripts/app/edopro_net_py.py host --deck decks/pool/snake-eye.ydk --port 8791 \
        --seed 70000 --agent-seed 101 --once --packet-log data/cache/app/host-client.jsonl.gz --no-discovery
    uv run python scripts/app/edopro_net_py.py join --deck decks/pool/branded.ydk --port 8791 \
        --agent-seed 202 --packet-log data/cache/app/py-client.jsonl.gz

The packet log is one gzip JSONL line per frame after framing: {"dir": "s"|"r", "p": proto, "x": payload hex}.
This file changes nothing under `src/`; it only subclasses/wraps the classes above.
"""

from __future__ import annotations

import argparse
import gzip
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks import load_ydk
from decklight.edopro import protocol as P
from decklight.edopro.bot import StreamBot
from decklight.edopro.client import ClientSession, DecisionOnlyView
from decklight.edopro.host import DuelHost, Session
from decklight.engine.duel import Engine

SPLITMIX_GOLDEN = 0x9E3779B97F4A7C15
M64 = (1 << 64) - 1


class StreamAgent:
    """The client-side splitmix64 agent of `bot.StreamBot` (act = next() % options, rps = 1 + next() % 3)."""

    name = "stream"
    needs = frozenset()

    def __init__(self, seed: int = 0) -> None:
        self.state = seed & M64

    def _next(self) -> int:
        self.state = (self.state + SPLITMIX_GOLDEN) & M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def rps(self) -> int:
        return 1 + self._next() % 3

    def go_first(self) -> bool:
        return True

    def act(self, view) -> int:
        return self._next() % len(view.opts)


class Log:
    def __init__(self, path: Path | None) -> None:
        self.f = gzip.open(path, "wt", encoding="utf-8") if path else None  # noqa: SIM115 - closed by close()
        self.n = 0

    def line(self, direction: str, proto: int, payload: bytes) -> None:
        if self.f is None:
            return
        self.f.write(json.dumps({"dir": direction, "p": proto, "x": payload.hex()}) + "\n")
        self.n += 1
        if self.n % 2048 == 0:  # keep it streaming without a gzip sync flush per packet
            self.f.flush()

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None


def patch_nodelay_host() -> None:
    """The oracle sockets do not disable Nagle; on loopback that costs ~40 ms per prompt. Bytes are unchanged."""
    orig_init = Session.__init__

    def init(self, host, conn):
        orig_init(self, host, conn)
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    Session.__init__ = init


def patch_nodelay_client() -> None:
    orig = ClientSession._connect

    def _connect(self):
        orig(self)
        try:
            if self.sock is not None:
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    ClientSession._connect = _connect


def patch_host(log: Log) -> None:
    orig_send = Session.send
    orig_recv = Session.recv_frames

    def send(self, proto: int, payload: bytes = b"") -> None:
        orig_send(self, proto, payload)
        log.line("s", proto, payload)

    def recv_frames(self):
        frames = orig_recv(self)
        for proto, payload in frames:
            log.line("r", proto, payload)
        return frames

    Session.send = send
    Session.recv_frames = recv_frames


def patch_client(log: Log, save_replay: Path | None) -> None:
    orig_send = ClientSession.send
    orig_wait = ClientSession._wait

    def send(self, proto: int, payload: bytes = b"") -> None:
        orig_send(self, proto, payload)
        log.line("s", proto, payload)

    def _wait(self):
        frames = orig_wait(self)
        if frames:
            for proto, payload in frames:
                log.line("r", proto, payload)
                if proto == P.STOC_NEW_REPLAY and save_replay is not None:
                    save_replay.parent.mkdir(parents=True, exist_ok=True)
                    save_replay.write_bytes(payload)
        return frames

    ClientSession.send = send
    ClientSession._wait = _wait


def _np_codes(path: Path) -> set[int]:
    import numpy as np

    with np.load(path) as d:
        return {int(c) for c in d["codes"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["host", "join"])
    ap.add_argument("--deck", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7911)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--name", default="decklight")
    ap.add_argument("--password", default="")
    ap.add_argument("--roomid", type=int, default=0)
    ap.add_argument("--hand", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None, help="deal seed (host)")
    ap.add_argument("--agent-seed", type=int, default=0)
    ap.add_argument("--bot-second", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--packet-log", type=Path, default=None)
    ap.add_argument("--save-replay", type=Path, default=None)
    ap.add_argument("--record-dir", type=Path, default=Path("data/cache/app/py_client"))
    ap.add_argument("--record-root", type=Path, default=Path("data/cache/app/py_host"))
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--no-discovery", action="store_true")
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--rule", type=int, default=3)
    ap.add_argument("--start-lp", type=int, default=8000)
    ap.add_argument("--room-notes", default="")
    a = ap.parse_args()

    log = Log(a.packet_log)
    deck = load_ydk(a.deck)
    engine = Engine()
    missing = [c for c in (*deck.main, *deck.extra) if c not in engine.cards]
    if missing:
        print(f"ALERT edopro_net_py: deck has {len(missing)} codes missing from the card DB", file=sys.stderr)
        return 2

    if a.mode == "host":
        patch_nodelay_host()
        patch_host(log)
        if a.no_discovery:
            DuelHost._discovery_loop = lambda self, stop, tcp_port: None
        bot = StreamBot(a.agent_seed)
        host = DuelHost(engine, deck, bot, seed=a.seed, bot_second=a.bot_second, name=a.name,
                        record_root=a.record_root, emb_codes=_np_codes(Path("data/cards/emb/qwen3-0.6b-d512.npz")))
        try:
            host.serve(a.bind, a.port, once=a.once)
        finally:
            log.close()
        return 0

    patch_nodelay_client()
    patch_client(log, a.save_replay)
    agent = StreamAgent(a.agent_seed)
    cards = engine.cards
    create = None
    if a.create:
        create = P.HostInfo(rule=a.rule, start_lp=a.start_lp)
    session = ClientSession(
        a.host, a.port, a.name, deck, agent,
        view_factory=lambda seat, codes: DecisionOnlyView(seat, codes, cards=cards),
        proto=P.ProtocolConfig(), password=a.password, gameid=a.roomid,
        hand=a.hand, timeout_s=a.timeout, record_dir=a.record_dir, create=create,
        room_notes=a.room_notes,
    )
    try:
        code = session.run()
    finally:
        log.close()
    print(json.dumps({"exit": code, "winner": session.record.winner,
                      "going_first": session.record.going_first,
                      "responses": session.record.n_responses,
                      "retries": session.record.retries}))
    return code


if __name__ == "__main__":
    sys.exit(main())
