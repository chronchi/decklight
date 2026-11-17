"""WindBot-style EDOPro network client: join a room and play one seat (plan §3.3).

`ClientSession` is a state machine (lobby -> RPS -> turn-order -> duel loop) that
talks to any EDOPro-protocol server (our `DuelHost`, EDOPro, Multirole): the room
host starts the duel, the bot only sends deck + ready, then answers its prompts.

The duel loop feeds every STOC_GAME_MSG payload to a view built by
`view_factory(seat, deck)` and sends one CTOS_RESPONSE per completed decision.
Only agents with `needs == frozenset()` work here (there is no local engine, so
nothing can call `obs()`); `check_needs` refuses the rest at startup.

Exit codes mirror `scripts/edopro_join.py`: 0 duel finished, 2 bad
args/server-side error, 3 view/server disagreement (MSG_RETRY), 4 timeout or
disconnect before the duel ended.
"""

from __future__ import annotations

import gzip
import json
import socket
import struct
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from decklight.edopro import protocol as P
from decklight.engine import constants as C
from decklight.engine import messages as M
from decklight.engine.actions import make_decision
from decklight.play.agents import check_needs

OK, BAD_ARGS, RETRY, TIMEOUT = 0, 2, 3, 4

DEFAULT_RECORD_DIR = Path("data/cache/edopro_client")


def _alert(record: dict, text: str) -> None:
    line = f"ALERT edopro_client: {text}"
    print(line, file=sys.stderr, flush=True)
    record.setdefault("alerts", []).append(line)


class DecisionOnlyView:
    """Minimal view for protocol-only agents (random): answers selects, nothing else.

    - `feed(payload)`: one STOC_GAME_MSG payload. Select messages for us become
      the pending decision (via `make_decision`); single-option decisions are
      answered internally (index 0, possibly over several steps) and never
      surface via `needs_answer`. MSG_WIN records the winner.
    - `needs_answer` / `opts` / `take(index) -> bytes | None`: the pending
      multi-option decision, one `choose` step at a time (`take` returns the
      response bytes once the decision completes, else None).
    - `take_auto() -> bytes | None`: next completed singleton response, if any.
    - `obs()` raises: there is no board here (needs `StreamView`).

    `pool` (for ANNOUNCE_CARD) is our deck list plus every card code seen on the
    wire; extra codes are harmless (any declarable code is a legal answer).
    """

    def __init__(self, seat: int, deck: list[int], cards=None, pool=None) -> None:
        self.seat = seat
        self.deck = list(deck)
        self._pool = set(pool) | set(deck) if pool else set(deck)
        self._cards = cards
        self._decision = None
        self._auto: list[bytes] = []
        self.winner: tuple[int, int] | None = None
        self.n_feed = 0

    def _ensure_cards(self):
        if self._cards is None:
            from decklight.engine.duel import Engine

            self._cards = Engine().cards
        return self._cards

    # -- feed --

    def feed(self, payload: bytes) -> None:
        if not payload:
            return
        self.n_feed += 1
        t = payload[0]
        if t == C.MSG_WIN:
            self.winner = (payload[1], payload[2]) if len(payload) >= 3 else None
            return
        if t == C.MSG_RETRY:
            return
        cards = self._ensure_cards()
        end = len(payload) - 3
        for i in range(end):
            code = int.from_bytes(payload[i : i + 4], "little")
            if code in cards:
                self._pool.add(code)
        if t not in M.SELECT_TYPES:
            return
        msg = M.parse_one(payload)
        if not isinstance(msg, M.Select):
            return
        dec = make_decision(msg, cards, lambda: self._pool)
        if len(dec.options()) <= 1:
            # Singleton: answer internally without involving the agent or the
            # record (the training semantics for auto-taken prompts).
            for _ in range(128):
                resp = dec.choose(0)
                if resp is not None:
                    self._auto.append(resp)
                    break
            else:
                raise RuntimeError(f"singleton decision never completes: {type(msg).__name__}")
        else:
            if self._decision is not None:
                raise RuntimeError("new select arrived while one is still pending")
            self._decision = dec

    # -- agent interface --

    @property
    def needs_answer(self) -> bool:
        return self._decision is not None

    @property
    def opts(self) -> list:
        return self._decision.options() if self._decision is not None else []

    @property
    def decision(self):
        return self._decision.msg if self._decision is not None else None

    def take(self, index: int) -> bytes | None:
        if self._decision is None:
            return None
        resp = self._decision.choose(index)
        if resp is not None:
            self._decision = None
        return resp

    def take_auto(self) -> bytes | None:
        if self._auto:
            return self._auto.pop(0)
        return None

    def obs(self) -> dict:
        raise NotImplementedError("needs StreamView")


# Game messages counted in DuelRecord.events, by card code.
_EVENT_KIND = {C.MSG_CHAINING: "chain", C.MSG_SUMMONING: "summon", C.MSG_SPSUMMONING: "summon",
               C.MSG_FLIPSUMMONING: "summon"}


_PHASES = {C.PHASE_DRAW: "Draw", C.PHASE_STANDBY: "Standby", C.PHASE_MAIN1: "Main 1",
           C.PHASE_BATTLE_START: "Battle", C.PHASE_MAIN2: "Main 2", C.PHASE_END: "End"}
_LP_MSGS = (C.MSG_DAMAGE, C.MSG_RECOVER, C.MSG_LPUPDATE, C.MSG_PAY_LPCOST)


def _card_name(code: int) -> str:
    try:
        from decklight.web.view import DB  # lazy: only the live log needs names
        return DB.names().get(code, "") or str(code)
    except Exception:  # names are a nicety; never break a duel over them
        return str(code)


class LiveLog:
    """One stderr line per turn, phase, LP change, activation/summon and our own pick (`LIVE mm:ss ...`).

    Follows a duel while it runs (the record is only written at the end). `who(p)` is "me"/"opp" from our seat.
    """

    def __init__(self, out=None) -> None:
        self.out = out if out is not None else sys.stderr
        self.t0 = time.monotonic()
        self.turn = 0
        self.lp: tuple[int, int] | None = None

    def line(self, text: str) -> None:
        m, sec = divmod(int(time.monotonic() - self.t0), 60)
        print(f"LIVE {m:02d}:{sec:02d} {text}", file=self.out, flush=True)

    def message(self, t: int, payload: bytes, seat: int | None, view) -> None:
        """Call after `view.feed(payload)`, so LP already reflects the message."""
        who = lambda p: "me" if p == seat else "opp"  # noqa: E731
        if t == C.MSG_NEW_TURN and len(payload) >= 2:
            self.turn += 1
            self.line(f"TURN {self.turn} ({who(payload[1] & 1)})")
        elif t == C.MSG_NEW_PHASE and len(payload) >= 3:
            name = _PHASES.get(struct.unpack_from("<H", payload, 1)[0])
            if name:
                self.line(f"  phase {name}")
        elif t in _EVENT_KIND and len(payload) >= 6:
            (code,) = struct.unpack_from("<I", payload, 1)
            self.line(f"  {who(payload[5])} {_EVENT_KIND[t]}s {_card_name(code)}")
        elif t == C.MSG_WIN and len(payload) >= 2:
            self.line(f"WIN {who(payload[1])} (reason {payload[2] if len(payload) > 2 else '?'})")
        if t in _LP_MSGS:
            lp = getattr(getattr(view, "state", None), "lp", None)
            if lp is not None and seat in (0, 1) and tuple(lp) != self.lp:
                self.lp = tuple(lp)
                self.line(f"  LP me {lp[seat]} / opp {lp[1 - seat]}")

    def pick(self, opts: list, index: int) -> None:
        if len(opts) < 2 or not 0 <= index < len(opts):
            return
        o = opts[index]
        card = getattr(o, "card", None)
        code = getattr(card, "code", 0) or getattr(o, "code", 0)
        kind = getattr(getattr(o, "kind", None), "name", "?").lower()
        self.line(f"    pick {kind}{' ' + _card_name(code) if code else ''} [{index + 1}/{len(opts)}]")


@dataclass
class DuelRecord:
    server: str = ""
    seat: int | None = None
    rps: int | None = None
    going_first: bool | None = None
    n_responses: int = 0
    turns: int = 0
    retries: int = 0
    winner: tuple[int, int] | None = None
    games: list = field(default_factory=list)  # one {"seat", "winner", "reason", "turns"} per duel of a match (Bo3)
    duel_end: bool = False
    packets: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)  # "chain:<code>" / "summon:<code>" counts, both players
    alerts: list = field(default_factory=list)
    players: dict = field(default_factory=dict)  # lobby seat -> name (STOC_HS_PLAYER_ENTER)
    chat: list = field(default_factory=list)
    replays: list = field(default_factory=list)  # `.yrpX` file names in record_dir, one per duel  # {"t", "from", "kind", "text"}, in arrival order
    exit_code: int = TIMEOUT


class ClientSession:
    """One duel as a bot: connect, lobby, RPS/TP, duel loop, record.

    By default the bot joins an existing room (`gameid`, `password`). With
    `create=HostInfo(...)` it creates the room instead (CTOS_CREATE_GAME, as an
    EDOPro player hosting on Multirole does) and starts the duel (CTOS_HS_START)
    once every other duelist is ready.
    """

    def __init__(
        self,
        host: str,
        port: int,
        name: str,
        deck,  # Deck (main/extra/side attrs) or (main, extra, side) tuple
        agent,
        view_factory: Callable[[int, list[int]], DecisionOnlyView],
        proto: P.ProtocolConfig | None = None,
        password: str = "",
        gameid: int = 0,
        hand: int = 0,  # forced RPS throw (WindBot Hand=); 0 = agent decides
        timeout_s: float = 30.0,
        record_dir: str | Path = DEFAULT_RECORD_DIR,
        create: P.HostInfo | None = None,
        room_notes: str = "",
        lobby_timeout_s: float | None = None,  # silence allowed before the duel starts (None: timeout_s)
        clock_cap_s: float | None = None,  # opponent to act: wait their clock + 30 s, at most this (None: timeout_s only)
    ) -> None:
        # Rooms with a game clock: while the server reports the opponent's clock (STOC_TIME_LIMIT), silence is allowed
        # until their last reported clock runs out + 30 s, capped (rooms with ~12 h clocks). Before the first report
        # (lobby, siding) the stall limit applies.
        self.clock_cap_s = clock_cap_s
        self.lobby_timeout_s = lobby_timeout_s
        self.host = host
        self.port = port
        self.name = name
        if hasattr(deck, "main"):
            self.main, self.extra, self.side = list(deck.main), list(deck.extra), list(deck.side)
        else:
            self.main, self.extra, self.side = (list(p) for p in deck)
        self.full_deck = self.main + self.extra + self.side
        self.agent = agent
        self.view_factory = view_factory
        self.proto = proto or P.ProtocolConfig()
        self.password = password
        self.gameid = gameid
        self.hand = hand
        self.timeout_s = timeout_s
        self.record_dir = Path(record_dir)
        self.sock: socket.socket | None = None
        self.reader = P.FrameReader()
        self.record = DuelRecord(server=f"{host}:{port}")
        self.on_end: Callable[[], None] | None = None  # called as soon as MSG_WIN/DUEL_END arrives
        self._ties = 0  # consecutive RPS ties (lockstep guard)
        self.create = create
        self.room_notes = room_notes
        self.room_id: int | None = None  # set from STOC_CREATE_GAME in create mode
        self.live = LiveLog()
        # every frame both ways, in order: (seconds, "in"/"out", proto, payload). Saved as <record>.stream.jsonl.gz
        # so a duel can be rebuilt from our side even when the server sends no replay (NOTEBOOK 2026-11-03).
        self.stream: list[tuple[float, str, int, bytes]] = []
        self._t0 = time.monotonic()

    # -- socket helpers --

    def _connect(self) -> None:
        deadline = time.monotonic() + min(self.timeout_s, 10.0)
        last: OSError | None = None
        while True:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
                break
            except OSError as e:
                last = e
                if time.monotonic() >= deadline:
                    raise last from None
                time.sleep(0.1)
        sock.settimeout(self.timeout_s)
        self.sock = sock

    def send(self, proto: int, payload: bytes = b"") -> None:
        assert self.sock is not None
        self.sock.sendall(P.pack_frame(proto, payload))
        self._count(f"C{proto:#x}")
        self.stream.append((round(time.monotonic() - self._t0, 3), "out", proto, payload))

    def _note_event(self, t: int, payload: bytes) -> None:
        """Count which cards were activated / summoned (what a duel record can be audited on)."""
        kind = _EVENT_KIND.get(t)
        if kind is None or len(payload) < 5:
            return
        (code,) = struct.unpack_from("<I", payload, 1)
        key = f"{kind}:{code}"
        self.record.events[key] = self.record.events.get(key, 0) + 1

    def _note_chat(self, proto: int, payload: bytes) -> None:
        """Keep every chat line (both packet kinds) in the record and echo it to stderr."""
        try:
            if proto == P.STOC_CHAT_2:
                kind, _team, who, text = P.parse_chat2(payload)
            else:
                player, text = P.parse_chat(payload)
                kind, who = "player", str(player)
        except ValueError as e:
            _alert(self.record.__dict__, f"bad chat packet: {e}")
            return
        line = {"t": round(time.time(), 1), "from": who, "kind": kind, "text": text}
        self.record.chat.append(line)
        print(f"CHAT [{kind}] {who}: {text}", file=sys.stderr, flush=True)

    def _save_stream(self, path: Path) -> str | None:
        """Write the frame log as gzip JSON lines: a header (deck, name, server), then {t, dir, proto, hex}."""
        if not self.stream:
            return None
        head = {"server": f"{self.host}:{self.port}", "name": self.name, "main": self.main, "extra": self.extra,
                "side": self.side, "pro_version": self.proto.pro_version,
                "client_version": list(self.proto.client_version)}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(head) + "\n")
            for t, d, proto, payload in self.stream:
                f.write(json.dumps({"t": t, "dir": d, "proto": proto, "hex": payload.hex()}) + "\n")
        return path.name

    def _save_replay(self, payload: bytes) -> None:
        """Keep the server's `.yrpX` (one per duel of a match) next to the record; `play/yrp.py` reads it."""
        if not payload:
            _alert(self.record.__dict__, "empty STOC_NEW_REPLAY")
            return
        try:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name)[:32]
            path = self.record_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}-{len(self.record.replays)}.yrpX"
            path.write_bytes(payload)
            self.record.replays.append(path.name)
        except OSError as e:
            _alert(self.record.__dict__, f"cannot write replay: {e}")

    def _count(self, key) -> None:
        self.record.packets[key] = self.record.packets.get(key, 0) + 1

    def _wait(self) -> list[tuple[int, bytes]] | None:
        """Next frames; None on timeout (code 4) or clean close (finish)."""
        assert self.sock is not None
        try:
            data = self.sock.recv(65536)
        except TimeoutError:
            return None
        except OSError:
            return []
        if not data:
            return []
        frames = self.reader.feed(data)
        t = round(time.monotonic() - self._t0, 3)
        self.stream.extend((t, "in", proto, payload) for proto, payload in frames)
        return frames

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.sock = None

    # -- run --

    def run(self) -> int:
        check_needs(self.agent, provides=frozenset(), where="EDOPro client")
        rec = self.record
        try:
            self._connect()
        except OSError as e:
            _alert(rec.__dict__, f"connect to {self.host}:{self.port} failed: {e}")
            return self._finish(TIMEOUT)
        self.send(P.CTOS_PLAYER_INFO, P.pack_player_info(self.name))
        if self.create is not None:
            self.send(P.CTOS_CREATE_GAME,
                      P.pack_create_game(self.create, self.name, self.password, self.room_notes))
        else:
            self.send(
                P.CTOS_JOIN_GAME,
                P.pack_join_game(
                    self.proto.pro_version,
                    self.gameid,
                    P.client_version_bytes(*self.proto.client_version),
                    self.password,
                ),
            )
        try:
            code, pending = self._lobby()
            if code is not None:
                return self._finish(code)
            return self._finish(self._duel(pending))
        finally:
            self.close()

    def _finish(self, code: int) -> int:
        self.record.exit_code = code
        try:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name)[:32]
            path = self.record_dir / f"{stamp}-{safe}-{self.port}.json"
            rec = dict(self.record.__dict__)
            if isinstance(rec["winner"], tuple):
                rec["winner"] = list(rec["winner"])
            rec["stream"] = self._save_stream(path.with_suffix(".stream.jsonl.gz"))
            path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
        except OSError as e:
            print(f"ALERT edopro_client: cannot write record: {e}", file=sys.stderr, flush=True)
        return code

    # -- lobby + RPS/TP --

    def _lobby(self) -> tuple[int | None, list[tuple[int, bytes]]]:
        """Lobby until STOC_DUEL_START.

        Returns (exit code, leftover frames) on rejection, or (None, frames
        batched after DUEL_START in the same read) to hand to `_duel`.
        """
        rec = self.record.__dict__
        deck_sent = False
        pending_ready = False
        seat: int | None = None
        is_host = False
        ready_sent = False
        others_ready: set[int] = set()  # create mode: seats of other duelists that are ready
        start_sent = False

        def maybe_start() -> None:
            nonlocal start_sent
            if self.create is not None and is_host and ready_sent and others_ready and not start_sent:
                self.send(P.CTOS_HS_START)
                start_sent = True
        if self.lobby_timeout_s is not None and self.sock is not None:
            self.sock.settimeout(self.lobby_timeout_s)  # an idle room host: give up, don't sit for timeout_s
        while True:
            frames = self._wait()
            if frames is None:
                _alert(rec, "lobby timed out waiting for the server")
                return TIMEOUT, []
            if not frames:
                _alert(rec, "server closed the connection in the lobby")
                return TIMEOUT, []
            for i, (proto, payload) in enumerate(frames):
                self._count(proto)
                if proto == P.STOC_ERROR_MSG:
                    _alert(rec, f"server rejected us: {P.parse_error_msg(payload)}")
                    return BAD_ARGS, []
                if proto == P.STOC_JOIN_GAME:
                    try:
                        info = P.parse_join_game(payload)
                    except ValueError as e:
                        _alert(rec, f"bad STOC_JOIN_GAME: {e}")
                        return BAD_ARGS, []
                    rec["server_version"] = list(info.version)
                    self.send(P.CTOS_UPDATE_DECK, P.pack_update_deck(self.main, self.extra, self.side))
                    deck_sent = True
                    if pending_ready:
                        self.send(P.CTOS_HS_READY)
                        pending_ready = False
                        ready_sent = True
                        maybe_start()
                elif proto == P.STOC_TYPE_CHANGE:
                    try:
                        seat, is_host = P.parse_type_change(payload)
                    except ValueError as e:
                        _alert(rec, f"bad STOC_TYPE_CHANGE: {e}")
                        return BAD_ARGS, []
                    rec["seat"] = seat
                    if seat == P.NETPLAYER_TYPE_OBSERVER:
                        # the duelist seats filled as we joined: we would only watch (NOTEBOOK 2026-11-03, g4)
                        _alert(rec, "joined as a spectator (no free duelist seat); leaving")
                        return BAD_ARGS, []
                    if deck_sent:
                        self.send(P.CTOS_HS_READY)
                        ready_sent = True
                        maybe_start()
                    else:
                        pending_ready = True
                elif proto == P.STOC_CREATE_GAME:
                    try:
                        self.room_id = P.parse_create_game(payload)
                    except ValueError as e:
                        _alert(rec, f"bad STOC_CREATE_GAME: {e}")
                        return BAD_ARGS, []
                    rec["room_id"] = self.room_id
                elif proto == P.STOC_HS_PLAYER_ENTER:
                    try:
                        _n, _p = P.parse_hs_player_enter(payload)
                        self.record.players[str(_p)] = _n
                    except ValueError:
                        pass
                elif proto == P.STOC_HS_PLAYER_CHANGE:
                    # a joiner ignores this (the room host starts the duel); in create mode we are the host
                    try:
                        pos, state = P.parse_hs_player_change(payload)
                    except ValueError:
                        continue
                    if pos != seat:
                        if state == "ready":
                            others_ready.add(pos)
                        else:
                            others_ready.discard(pos)
                        maybe_start()
                elif proto == P.STOC_SELECT_HAND:
                    rps = self.hand or self.agent.rps()
                    if rps not in (1, 2, 3):
                        _alert(rec, f"illegal RPS throw {rps}, sending 1")
                        rps = 1
                    self.record.rps = rps
                    self.send(P.CTOS_HAND_RESULT, P.pack_hand_choice(rps))
                elif proto == P.STOC_HAND_RESULT:
                    if self._note_hand_result(payload):
                        _alert(rec, "50 consecutive RPS ties "
                                    "(both sides use the same seed?)")
                        return RETRY, []
                elif proto == P.STOC_SELECT_TP:
                    first = bool(self.agent.go_first())
                    self.record.going_first = first
                    self.send(P.CTOS_TP_RESULT, P.pack_tp_choice(first))
                elif proto == P.STOC_TP_RESULT:
                    pass
                elif proto == P.STOC_DUEL_START:
                    self.record.seat = seat
                    self._lobby_seat = seat
                    return None, frames[i + 1 :]
                elif (proto == P.STOC_GAME_MSG and deck_sent
                      and payload[:1] == bytes([C.MSG_START])):
                    # ygo-windbot starts a WindBot game with MSG_START and no STOC_DUEL_START;
                    # the duel loop reads our seat from that message.
                    self._lobby_seat = seat
                    return None, frames[i:]
                elif proto == P.STOC_DUEL_END:
                    self.record.duel_end = True
                    return OK, []
                elif proto in (P.STOC_CHAT, P.STOC_CHAT_2):
                    self._note_chat(proto, payload)
                elif proto in (P.STOC_REPLAY, P.STOC_HS_WATCH_CHANGE):
                    pass  # safely ignored
                # STOC_GAME_MSG / unknown: ignored before the duel starts.

    # -- duel --

    def _note_hand_result(self, payload: bytes) -> bool:
        """Record a STOC_HAND_RESULT; True when the RPS tie-loop guard trips.

        Two equal seeds throw identical sequences and tie forever
        (p(tie)^50 ~ 1e-24 by chance); fail fast instead of hanging.
        """
        try:
            ours, theirs = P.parse_hand_result(payload)
        except ValueError:
            return False
        self.record.__dict__["hand_result"] = [ours, theirs]
        if ours == theirs:
            self._ties += 1
            return self._ties >= 50
        self._ties = 0
        return False

    def _duel(self, pending: list[tuple[int, bytes]]) -> int:
        rec = self.record.__dict__
        if self.sock is not None:
            self.sock.settimeout(self.timeout_s)
        seat: int | None = getattr(self, "_lobby_seat", None)
        view: DecisionOnlyView | None = None
        queue: list[tuple[int, bytes]] | None = list(pending) or None
        while True:
            if queue is None:
                frames = self._wait()
                if frames is None:
                    _alert(rec, "timed out waiting for the server mid-duel")
                    return TIMEOUT
                if not frames:
                    # Socket close ends the session; a MSG_WIN already ended the duel.
                    if view is not None and view.winner is not None:
                        return OK
                    _alert(rec, "server closed the connection mid-duel")
                    return OK if self.record.duel_end else TIMEOUT
            else:
                frames, queue = queue, None
            for proto, payload in frames:
                self._count(proto)
                if proto == P.STOC_ERROR_MSG:
                    _alert(rec, f"server error mid-duel: {P.parse_error_msg(payload)}")
                    return BAD_ARGS
                if proto == P.STOC_GAME_MSG:
                    if not payload:
                        continue
                    t = payload[0]
                    if t == C.MSG_START:
                        self._duels = getattr(self, "_duels", 0) + 1
                        seat = payload[1] & 0xF if len(payload) >= 2 else seat
                        self.record.seat = seat
                        view = self.view_factory(seat if seat is not None else 0, self.full_deck)
                        view.feed(payload)  # deck and extra counts: without it the opponent's deck read 0 all duel
                        continue
                    if t == C.MSG_RETRY:
                        self.record.retries += 1
                        _alert(rec, "MSG_RETRY: our response disagrees with the server")
                        return RETRY
                    if view is None:
                        continue  # refresh before START: nothing to answer
                    if t == C.MSG_NEW_TURN:
                        self.record.turns += 1
                    elif t == C.MSG_WIN and len(payload) >= 3:
                        self._win_reason = payload[2]  # StreamView keeps only the winner
                    self._note_event(t, payload)
                    view.feed(payload)
                    self.live.message(t, payload, seat, view)
                    while True:
                        auto = view.take_auto()
                        if auto is None:
                            break
                        self.send(P.CTOS_RESPONSE, P.pack_response(auto))
                        self.record.n_responses += 1
                    while view.needs_answer:
                        index = self.agent.act(view)
                        self.live.pick(view.opts, index)
                        resp = view.take(index)
                        if resp is not None:
                            self.send(P.CTOS_RESPONSE, P.pack_response(resp))
                            self.record.n_responses += 1
                    if view.winner is not None:
                        if len(self.record.games) < getattr(self, "_duels", 1):
                            w, why = (view.winner if isinstance(view.winner, tuple)
                                      else (view.winner, getattr(self, "_win_reason", None)))
                            self.record.games.append({"seat": seat, "winner": w, "reason": why, "turns": self.record.turns})
                        self.record.winner = view.winner
                        if self.on_end is not None:
                            self.on_end()
                            self.on_end = None
                elif proto == P.STOC_REMATCH:
                    # one duel per session: decline, the server then ends the room (STOC_DUEL_END)
                    self.send(P.CTOS_REMATCH_RESPONSE, bytes([0]))
                elif proto == P.STOC_SELECT_HAND:
                    rps = self.hand or self.agent.rps()
                    if rps not in (1, 2, 3):
                        rps = 1
                    self.record.rps = rps
                    self.send(P.CTOS_HAND_RESULT, P.pack_hand_choice(rps))
                elif proto == P.STOC_SELECT_TP:
                    first = bool(self.agent.go_first())
                    self.record.going_first = first
                    self.send(P.CTOS_TP_RESULT, P.pack_tp_choice(first))
                elif proto == P.STOC_HAND_RESULT:
                    if self._note_hand_result(payload):
                        _alert(rec, "50 consecutive RPS ties "
                                    "(both sides use the same seed?)")
                        return RETRY
                elif proto == P.STOC_TIME_LIMIT:
                    try:
                        player, secs = P.parse_time_limit(payload)
                    except ValueError:
                        continue
                    if player == (self.record.seat if self.record.seat is not None else seat):
                        self.send(P.CTOS_TIME_CONFIRM, P.pack_time_confirm())
                    elif self.clock_cap_s is not None and self.sock is not None:
                        # the server reports the opponent's clock only now and then (seen once, then 120 s of
                        # silence after our answer), so their last reported clock holds until the next report
                        self.sock.settimeout(max(self.timeout_s, min(secs + 30.0, self.clock_cap_s)))
                elif proto == P.STOC_CHANGE_SIDE:
                    # Match siding: resend the same deck.
                    self.send(P.CTOS_UPDATE_DECK, P.pack_update_deck(self.main, self.extra, self.side))
                elif proto == P.STOC_DUEL_END:
                    self.record.duel_end = True
                    if view is not None and view.winner is not None:
                        self.record.winner = view.winner
                    if self.on_end is not None:
                        self.on_end()
                        self.on_end = None
                    return OK
                elif proto in (P.STOC_CHAT, P.STOC_CHAT_2):
                    self._note_chat(proto, payload)
                elif proto == P.STOC_NEW_REPLAY:
                    self._save_replay(payload)
                elif proto in (
                    P.STOC_TP_RESULT, P.STOC_REPLAY,
                    P.STOC_HS_PLAYER_ENTER, P.STOC_HS_PLAYER_CHANGE, P.STOC_HS_WATCH_CHANGE,
                    P.STOC_CREATE_GAME, P.STOC_JOIN_GAME, P.STOC_TYPE_CHANGE,
                    P.STOC_LEAVE_GAME, P.STOC_DUEL_START, P.STOC_WAITING_SIDE,
                    P.STOC_WAITING_REMATCH,
                ):
                    pass  # counted, safely ignored
