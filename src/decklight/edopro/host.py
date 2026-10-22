"""Local EDOPro-compatible host (1v1): our engine runs the duel, a human joins from EDOPro.

Flow (port of generic_duel.cpp for one human + one local bot, no observers/match/tag):
player info -> join (our HostInfo echoes the client's version2, so any client matches)
-> deck -> ready -> RPS -> TP -> duel (filtered STOC_GAME_MSG for the human,
MSG_WAITING while the bot thinks) -> win -> STOC_DUEL_END -> record ->
back to accept (unless --once).

The authoritative duel is a PyEnv. The human's raw responses are decoded by
decklight.engine.inverse (D1; imported lazily) into env option indices; the bot
acts through the Bot interface (bot.py).
"""

import os
import random
import socket
import struct
import threading
from pathlib import Path

import numpy as np

from decklight.edopro import filter as F
from decklight.edopro import protocol as P
from decklight.edopro.bot import Bot
from decklight.engine import constants as C
from decklight.engine.duel import TCG_FLAGS, DuelConfig, Engine
from decklight.engine.env import PyEnv
from decklight.play.session import (
    DriverDiverged,
    DuelResult,
    DuelSession,
)
from decklight.play.session import (
    write_record as _write_record,
)

EMB_PATH = Path(os.environ.get("YGO_EMB") or
               Path(__file__).resolve().parents[3] / "data" / "cards" / "emb" / "qwen3-0.6b-d512.npz")

EXTRA_TYPES = C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ


def split_deck(codes: list[int], cards: dict) -> tuple[list[int], list[int]]:
    """UpdateDeck mainc block -> (main, extra) by card type (rituals stay main)."""
    main, extra = [], []
    for c in codes:
        t = cards[c].type
        if t & EXTRA_TYPES or (t & C.TYPE_LINK and t & C.TYPE_MONSTER):
            extra.append(c)
        else:
            main.append(c)
    return main, extra


def check_deck(main: list[int], extra: list[int], cards: dict, emb_codes: set[int],
               ) -> tuple[int, int, int, int, int] | None:
    """DeckError args (type, current, minimum, maximum, code) or None when legal."""
    if not 40 <= len(main) <= 60:
        return (P.DERR_MAINCOUNT, len(main), 40, 60, 0)
    if len(extra) > 15:
        return (P.DERR_EXTRACOUNT, len(extra), 0, 15, 0)
    for c in (*main, *extra):
        if c not in cards or c not in emb_codes:
            return (P.DERR_UNKNOWNCARD, 0, 0, 0, c)
    return None


class Disconnect(Exception):
    pass


class Surrender(Exception):
    pass


# DuelResult / DriverDiverged live in decklight.play.session (imported above);
# they stay importable from here (`edopro.host.DuelResult`).


class Session:
    """One TCP connection: lobby, RPS/TP, then the duel."""

    def __init__(self, host: "DuelHost", conn: socket.socket) -> None:
        self.host = host
        self.conn = conn
        self.reader = P.FrameReader()
        self.name = ""
        self.human_main: list[int] = []
        self.human_extra: list[int] = []
        self.human_side: list[int] = []
        self.has_deck = False
        self.ready = False
        self.started = False
        self.human_seat = 0
        self.bot_seat = 1
        self.env: PyEnv | None = None
        self.duel: DuelSession | None = None
        self.sent_trace = 0
        self.completed_duel = False
        self.product_info: dict | None = None

    # --- socket helpers ---

    def send(self, proto: int, payload: bytes = b"") -> None:
        self.conn.sendall(P.pack_frame(proto, payload))

    def recv_frames(self) -> list[tuple[int, bytes]]:
        try:
            data = self.conn.recv(65536)
        except (ConnectionError, OSError):
            raise Disconnect from None
        if not data:
            raise Disconnect
        return self.reader.feed(data)

    # --- lobby ---

    def run(self) -> None:
        try:
            while not self.started:
                for proto, payload in self.recv_frames():
                    self.on_lobby(proto, payload)
        except (Disconnect, Surrender):
            pass

    def on_lobby(self, proto: int, payload: bytes) -> None:
        if proto == P.CTOS_PLAYER_INFO:
            self.name = P.decode_name(payload[:40]) or "human"
        elif proto == P.CTOS_JOIN_GAME:
            if len(payload) < 50:
                return
            _ver, _gameid, version2 = P.unpack_join_game(payload)
            info = self.host.host_info(version2)
            self.send(P.STOC_HS_PLAYER_ENTER, P.encode_name(self.name) + bytes([0]))
            self.send(P.STOC_JOIN_GAME, info.pack())
            self.send(P.STOC_TYPE_CHANGE, bytes([0x10]))  # pos 0 + host bit
        elif proto == P.CTOS_UPDATE_DECK:
            try:
                mixed, side = P.parse_update_deck(payload)
            except ValueError:
                self.send(P.STOC_ERROR_MSG, P.pack_deck_error(P.DERR_INVALIDSIZE))
                return
            try:
                main, extra = split_deck(mixed, self.host.engine.cards)
            except KeyError as e:
                self.send(P.STOC_ERROR_MSG, P.pack_deck_error(P.DERR_UNKNOWNCARD, code=int(e.args[0])))
                return
            self.human_main, self.human_extra, self.human_side = main, extra, side
            self.has_deck = True
        elif proto == P.CTOS_HS_READY:
            self.player_ready()
        elif proto == P.CTOS_HS_NOTREADY:
            self.ready = False
            self.send(P.STOC_HS_PLAYER_CHANGE, bytes([(0 << 4) | P.PLAYERCHANGE_NOTREADY]))
        elif proto == P.CTOS_HS_START:
            if self.ready and not self.started:
                self.start_duel()
        elif proto == P.CTOS_LEAVE_GAME:
            raise Disconnect
        # CTOS_CHAT / CTOS_TIME_CONFIRM / observer switches: ignored (no observers).

    def player_ready(self) -> None:
        if not self.has_deck:
            err = (P.DERR_MAINCOUNT, 0, 40, 60, 0)
        else:
            err = check_deck(self.human_main, self.human_extra,
                             self.host.engine.cards, self.host.emb_codes)
        if err is not None:
            self.send(P.STOC_HS_PLAYER_CHANGE, bytes([(0 << 4) | P.PLAYERCHANGE_NOTREADY]))
            self.send(P.STOC_ERROR_MSG, P.pack_deck_error(*err))
            return
        self.ready = True
        self.send(P.STOC_HS_PLAYER_CHANGE, bytes([(0 << 4) | P.PLAYERCHANGE_READY]))
        self.start_duel()

    # --- RPS / TP ---

    def await_packet(self, want: set[int]) -> tuple[int, bytes]:
        """Next frame whose proto is in `want` (surrender/leave raise)."""
        while True:
            for proto, payload in self.recv_frames():
                if proto == P.CTOS_SURRENDER:
                    raise Surrender
                if proto == P.CTOS_LEAVE_GAME:
                    raise Disconnect
                if proto in want:
                    return proto, payload

    def rps_tp(self) -> tuple[int, int]:
        """RPS then TP choice. Returns (human_seat, bot_seat)."""
        rps_bot = self.host.bot_rps()
        while True:
            self.send(P.STOC_SELECT_HAND)
            _, payload = self.await_packet({P.CTOS_HAND_RESULT})
            rps_human = payload[0] if payload else 0
            if rps_human not in (1, 2, 3):
                continue  # invalid: re-prompt (generic_duel ignores it and waits)
            self.send(P.STOC_HAND_RESULT, P.pack_hand_result(rps_human, rps_bot))
            if rps_human == rps_bot:
                rps_bot = self.host.bot_rps()
                continue
            human_won = P.rps_beats(rps_human, rps_bot)
            break
        if human_won:
            self.send(P.STOC_SELECT_TP)
            _, payload = self.await_packet({P.CTOS_TP_RESULT})
            human_first = (payload[0] if payload else 0) == 1
        else:
            human_first = self.host.bot_second
        return (0, 1) if human_first else (1, 0)

    # --- duel ---

    def start_duel(self) -> None:
        self.started = True
        self.send(P.STOC_DUEL_START)
        try:
            self.human_seat, self.bot_seat = self.rps_tp()
        except (Disconnect, Surrender):
            return
        seed = self.host.deal_seed()
        print(f"deal seed {seed} (human seat {self.human_seat})", flush=True)
        human_deck = {"main": tuple(self.human_main), "extra": tuple(self.human_extra)}
        first = human_deck if self.human_seat == 0 else self.host.bot_deck
        second = self.host.bot_deck if self.human_seat == 0 else human_deck
        cfg = DuelConfig(_DeckLike(first), _DeckLike(second), seed=seed)
        # auto_single=False: every core prompt is a real decision for someone. With the
        # default True the env would silently answer single-option prompts (empty chains,
        # forced picks) that we already forwarded to the human, desyncing the stream.
        launcher = self.host.bot if hasattr(self.host.bot, "start_duel") else None
        bot = None if launcher is not None else self.host.bot
        self.duel = DuelSession(
            self.host.engine, cfg, self.human_seat, bot=bot, launcher=launcher,
            seed=seed, human_side=self.human_side, record_root=self.host.record_root)
        self.duel.reset()
        self.env = self.duel.env
        self.sent_trace = 0
        driver = self.duel.driver
        self.product_info = self.duel.product_info
        n0, e0 = len(cfg.deck0.main), len(cfg.deck0.extra)
        n1, e1 = len(cfg.deck1.main), len(cfg.deck1.extra)
        self.send(P.STOC_GAME_MSG, P.build_start(cfg.lp, n0, e0, n1, e1, self.human_seat))
        for seat in (0, 1):
            self.send_refresh(("loc", seat, C.LOCATION_EXTRA))
        try:
            if driver is not None:
                result = self.run_duel_driver(cfg, seed, driver)
            else:
                result = self.run_duel(cfg, seed)
        except Surrender:
            winner = 1 - self.human_seat
            self.send(P.STOC_GAME_MSG, bytes([C.MSG_WIN, winner, 0]))
            env = self.env
            result = DuelResult(winner, env.duel.turn, list(env.choices),
                                [e.actor for e in env.history], surrendered=True)
        except Disconnect:
            env = self.env
            result = DuelResult(self.bot_seat, env.duel.turn, list(env.choices),
                                [e.actor for e in env.history], disconnected=True)
            self.host.write_record(cfg, seed, result, self)
            raise
        except DriverDiverged as e:
            print(f"ALERT edopro_host: product driver diverged: {e}", flush=True)
            env = self.env
            result = DuelResult(None, env.duel.turn, list(env.choices),
                                [e.actor for e in env.history], diverged=True)
        finally:
            if driver is not None:
                driver.close()
        self.send(P.STOC_DUEL_END)
        self.host.write_record(cfg, seed, result, self)
        self.completed_duel = True

    def send_refresh(self, ref: tuple) -> None:
        assert self.env is not None
        duel = self.env.duel
        if ref[0] == "loc":
            _, player, location = ref
            cards = duel.query_location(player, location)
            public = self.human_seat != player
            self.send(P.STOC_GAME_MSG, F.encode_update_data(player, location, cards, public))
        else:
            _, player, location, seq = ref
            cards = duel.query_location(player, location)
            if seq >= len(cards) or cards[seq] is None:
                return
            public = self.human_seat != player
            self.send(P.STOC_GAME_MSG,
                      F.encode_update_card(player, location, seq, cards[seq], public))

    def pump(self) -> None:
        """Send every new core message (filtered) with its before/after refreshes."""
        assert self.env is not None
        duel = self.env.duel
        new = duel.trace[self.sent_trace :]
        self.sent_trace = len(duel.trace)
        for raw in new:
            for ref in F.before_refreshes(raw):
                self.send_refresh(ref)
            msg = F.filtered_message(raw, self.human_seat)
            if msg is not None:
                self.send(P.STOC_GAME_MSG, msg)
            for ref in F.after_refreshes(raw):
                self.send_refresh(ref)

    def run_duel_driver(self, cfg: DuelConfig, seed: int, driver) -> DuelResult:
        """The product duel: bot choices come from the driver, human choices go to both envs.

        Lockstep: after every non-singleton step the host env's (player, option count)
        must equal the driver's outer decision log; singletons the product skipped are
        auto-taken (index 0) for the bot and dropped from the human feed. At the end the
        host history (mapped to product space) and the winner must equal the driver's.
        Stepping and lockstep live in DuelSession; this keeps the socket feed.
        """
        from decklight.engine.inverse import choices_for_response  # D1; fail fast if missing

        assert self.env is not None and self.duel is not None
        env = self.env
        while not env.done:
            self.pump()
            if env.done:
                break
            if env.player == self.bot_seat:
                if len(env.opts) == 1:
                    env.step(0)  # singleton the product's CppEnv took automatically
                    continue
                self.send(P.STOC_GAME_MSG, bytes([C.MSG_WAITING]))
                self.duel.step_bot()
            else:
                _, payload = self.await_packet({P.CTOS_RESPONSE})
                indices = choices_for_response(env, payload)
                if indices is None:
                    self.send(P.STOC_GAME_MSG, bytes([C.MSG_RETRY]))
                    continue
                for i in indices:
                    self.duel.step_human(i)
        self.pump()
        return self.duel.finish()

    def run_duel(self, cfg: DuelConfig, seed: int) -> DuelResult:
        from decklight.engine.inverse import choices_for_response  # D1; fail fast if missing

        assert self.env is not None and self.duel is not None
        env = self.env
        while not env.done:
            self.pump()
            if env.done:
                break
            if env.player == self.bot_seat:
                self.send(P.STOC_GAME_MSG, bytes([C.MSG_WAITING]))
                self.duel.step_bot()
            else:
                if os.environ.get("EDOPRO_DEBUG"):
                    print(f"ASK t={env.dec.msg.type} actor={env.player} step={env.dec.step}",
                          flush=True)
                _, payload = self.await_packet({P.CTOS_RESPONSE})
                if os.environ.get("EDOPRO_DEBUG"):
                    print(f"GOT resp={payload.hex()}", flush=True)
                indices = choices_for_response(env, payload)
                if indices is None:
                    if os.environ.get("EDOPRO_DEBUG"):
                        print(f"RETRY msg={env.dec.msg} resp={payload.hex()} "
                              f"opts={len(env.opts)} step={env.dec.step}", flush=True)
                    self.send(P.STOC_GAME_MSG, bytes([C.MSG_RETRY]))
                    continue
                for i in indices:
                    env.step(i)
        self.pump()
        return DuelResult(env.duel.winner, env.duel.turn, list(env.choices),
                          [e.actor for e in env.history])


class _DeckLike:
    def __init__(self, d) -> None:
        self.main = list(d["main"] if isinstance(d, dict) else d.main)
        self.extra = list(d["extra"] if isinstance(d, dict) else d.extra)


class DuelHost:
    """Listens for one human at a time; the bot holds the other seat."""

    def __init__(self, engine: Engine, bot_deck, bot: Bot, *, seed: int | None = None,
                 bot_second: bool = False, name: str = "ygo-zero",
                 record_root: Path = Path("runs/edopro")) -> None:
        self.engine = engine
        self.bot_deck = bot_deck
        self.bot = bot
        self.base_seed = seed
        self.bot_second = bot_second
        self.name = name
        self.record_root = record_root
        self._rng = random.Random()
        self.emb_codes = {int(c) for c in np.load(EMB_PATH)["codes"]}
        self._duels = 0

    def host_info(self, version2: bytes) -> P.HostInfo:
        flags = TCG_FLAGS
        return P.HostInfo(
            version=bytes(version2),
            start_lp=8000, start_hand=5, draw_count=1,
            duel_flag_low=flags & 0xFFFFFFFF, duel_flag_high=(flags >> 32) & 0xFFFFFFFF,
        )

    def bot_rps(self) -> int:
        rps = getattr(self.bot, "rps", None)
        return rps() if callable(rps) else self._rng.randint(1, 3)

    def deal_seed(self) -> int:
        return self.base_seed if self.base_seed is not None else self._rng.getrandbits(63)

    def serve(self, bind: str, port: int, once: bool = False) -> None:
        stop = threading.Event()
        udp = threading.Thread(target=self._discovery_loop, args=(stop, port), daemon=True)
        udp.start()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((bind, port))
        srv.listen(1)
        print(f"ygo-zero host on {bind or '0.0.0.0'}:{port} (room '{self.name}'); "
              f"join via LAN or 127.0.0.1:{port}", flush=True)
        try:
            while True:
                conn, addr = srv.accept()
                print(f"connection from {addr}", flush=True)
                try:
                    Session(self, conn).run()
                except (Disconnect, ConnectionError, OSError) as e:
                    print(f"session ended: {e}", flush=True)
                finally:
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    conn.close()
                if once and self._duels > 0:
                    break
        finally:
            stop.set()
            srv.close()

    def _discovery_loop(self, stop: threading.Event, tcp_port: int) -> None:
        """Answer EDOPro LAN HostRequests (UDP 7920 -> reply to sender:7921)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", P.DISCOVERY_PORT))
            sock.settimeout(0.5)
        except OSError as e:
            print(f"UDP discovery disabled: {e}", flush=True)
            return
        with sock:
            while not stop.is_set():
                try:
                    data, addr = sock.recvfrom(256)
                except TimeoutError:
                    continue
                except OSError:
                    return
                if len(data) >= 2 and struct.unpack_from("<H", data)[0] == P.NETWORK_CLIENT_ID:
                    pkt = (struct.pack("<HHHI", P.NETWORK_SERVER_ID, P.PRO_VERSION, tcp_port, 0)
                           + P.encode_name(self.name)
                           + self.host_info(P.client_version_bytes(*P.CLIENT_VERSION)).pack())
                    try:
                        sock.sendto(pkt, (addr[0], P.DISCOVERY_REPLY_PORT))
                    except OSError:
                        pass

    def write_record(self, cfg: DuelConfig, seed: int, result: DuelResult, session: Session) -> Path:
        bot_seat = 1 - session.human_seat
        bot_deck = cfg.deck0 if bot_seat == 0 else cfg.deck1
        path = _write_record(
            self.record_root, cfg, seed, result, human_seat=session.human_seat,
            human_side=session.human_side,
            bot_name=self.deck_name(bot_deck, bot_seat, session),
            product_info=getattr(session, "product_info", None))
        self._duels += 1
        return path

    def deck_name(self, deck, seat: int, session: Session) -> str:
        if seat == session.human_seat:
            return "human"
        name = getattr(self.bot_deck, "name", "") or "bot"
        return name
