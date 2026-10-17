"""WebSocket game server: play the product from a phone browser (W1).

One game at a time per server; a new connection takes over the session (the
phone reconnecting). The duel itself is a `play.DuelSession` — the same
transport-independent session behind the EDOPro host — with the human's
single-option prompts auto-answered and every multi-option prompt sent to the
client. The bot runs in its thread (`asyncio.to_thread`); while it thinks, a
`state` goes out every 1 s with `thinking.seconds`.

Protocol: see docs/plans/2026-10-17-web-client.md (the contract W1/W2 share).
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
import urllib.request
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

ROOT = Path(__file__).resolve().parents[3]
DECK_DIRS = (("pool", ROOT / "decks" / "pool"), ("newcards", ROOT / "decks" / "newcards"))
PICS_DIR = ROOT / "data" / "cache" / "pics"

IMG_URLS = {
    "small": "https://images.ygoprodeck.com/images/cards_small/{code}.jpg",
    "full": "https://images.ygoprodeck.com/images/cards/{code}.jpg",
}

PLACEHOLDER = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ygo-zero web client</title></head><body>
<h1>ygo-zero web client</h1>
<p>The phone client (W2) is not built yet. The game protocol lives at <code>/ws</code>.</p>
</body></html>
"""

CARD_BACK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="146" viewBox="0 0 100 146">
<rect width="100" height="146" rx="6" fill="#1d2a5e"/>
<rect x="8" y="8" width="84" height="130" rx="4" fill="none" stroke="#c8a24a" stroke-width="2"/>
<circle cx="50" cy="73" r="22" fill="none" stroke="#c8a24a" stroke-width="2"/>
<text x="50" y="77" text-anchor="middle" font-size="10" fill="#c8a24a">ygo-zero</text>
</svg>
"""


def _fetch_img(url: str, timeout: float = 5.0) -> bytes:
    """One image fetch (module-level so tests can monkeypatch it)."""
    req = urllib.request.Request(url, headers={"User-Agent": "ygo-zero-web/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _pretty_deck_name(stem: str) -> str:
    return re.sub(r"-meta\d+$", "", stem).replace("-", " ").title()


class Server:
    """App state: decks, the engine, the opponent factory, and the live game."""

    def __init__(self, *, token: str, engine=None, make_opponent=None,
                 record_root: Path = ROOT / "runs" / "web",
                 pics_dir: Path = PICS_DIR, opponent_default: str = "product",
                 device: str = "cpu", replay_threads: int = 1) -> None:
        self.token = token
        self._engine = engine
        self._make_opponent = make_opponent or self._default_opponent
        self.record_root = record_root
        self.pics_dir = pics_dir
        self.opponent_default = opponent_default
        self.device = device
        self.replay_threads = replay_threads
        self._emb_codes: set[int] | None = None
        self.lock = asyncio.Lock()
        self.ws: WebSocket | None = None
        self.session = None  # DuelSession | None
        self.mode = opponent_default
        self.human_name = "human"
        self.prompt_id = 0
        self.log_sent = 0
        self.gen = 0
        self.finished = False
        self.end_reason = ""
        self._last_sig: tuple = ()
        self.user_decks: dict[str, str] = {}  # id -> ydk text (pasted this session)

    # --- lazy resources ---

    @property
    def engine(self):
        if self._engine is None:
            from ygo_zero.engine.duel import Engine

            self._engine = Engine()
        return self._engine

    def emb_codes(self) -> set[int]:
        if self._emb_codes is None:
            import numpy as np

            from ygo_zero.edopro.host import EMB_PATH

            self._emb_codes = {int(c) for c in np.load(EMB_PATH)["codes"]}
        return self._emb_codes

    def _default_opponent(self, mode: str):
        from ygo_zero.edopro.bot import RandomBot
        from ygo_zero.edopro.product_bot import ProductLauncher

        if mode == "random":
            return RandomBot(random.Random().randrange(1 << 30))
        if mode == "fast":
            return ProductLauncher(device=self.device, fast=True,
                                   replay_threads=self.replay_threads)
        return ProductLauncher(device=self.device, replay_threads=self.replay_threads)

    # --- decks ---

    def list_decks(self) -> list[dict]:
        out, seen = [], set()
        for group, d in DECK_DIRS:
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.ydk")):
                if p.stem in seen:
                    continue
                seen.add(p.stem)
                out.append({"id": p.stem, "name": _pretty_deck_name(p.stem),
                            "group": group})
        for i, (deck_id, _text) in enumerate(sorted(self.user_decks.items())):
            out.append({"id": deck_id, "name": f"Pasted deck {i + 1}", "group": "user"})
        return out

    def resolve_deck(self, spec):
        """A deck id or {"ydk": text} -> (Deck, error text|None)."""
        from ygo_zero.decks import load_ydk, parse_ydk

        if isinstance(spec, dict) and "ydk" in spec:
            deck = parse_ydk(spec["ydk"], "pasted")
            return deck, self.check_text(deck)
        if isinstance(spec, str):
            for _group, d in DECK_DIRS:
                p = d / f"{spec}.ydk"
                if p.is_file():
                    deck = load_ydk(p)
                    return deck, self.check_text(deck)
            if spec in self.user_decks:
                deck = parse_ydk(self.user_decks[spec], spec)
                return deck, self.check_text(deck)
            return None, f"unknown deck {spec!r}"
        return None, "deck must be a deck id or {\"ydk\": \"...\"}"

    def check_text(self, deck) -> str | None:
        """The EDOPro host's deck rules; names cards without embeddings."""
        from ygo_zero.demos.names import CardNames
        from ygo_zero.edopro.host import check_deck, split_deck

        main, extra = split_deck([*deck.main, *deck.extra], self.engine.cards)
        err = check_deck(main, extra, self.engine.cards, self.emb_codes())
        if err is None:
            return None
        type_, cur, minimum, maximum, code = err
        names = CardNames().names
        extra_txt = ""
        if code:
            extra_txt = f": {code} ({names.get(code, 'unknown card')})"
        kinds = {0: "main-deck count", 1: "extra-deck count", 2: "unknown card"}
        what = kinds.get(type_, f"deck error {type_}")
        return (f"illegal deck ({what}: have {cur}, allowed {minimum}-{maximum})"
                f"{extra_txt}")

    # --- state ---

    def thinking_seconds(self) -> float | None:
        s = self.session
        if s is not None and s.thinking_since is not None and not s.done:
            return time.monotonic() - s.thinking_since
        return None

    def state_message(self, *, full_log: bool = False, peek: bool = False) -> dict:
        from ygo_zero.web.view import build_events, build_prompt, build_view

        s = self.session
        if s is None:
            return {"t": "lobby", "decks": self.list_decks()}
        human = s.human_seat
        view = build_view(s.env, human)
        prompt = None
        if s.need_human and not s.done:
            self.prompt_id = s.current_prompt_id()
            prompt = build_prompt(s.env, human, self.prompt_id)
        thinking = self.thinking_seconds()
        log = build_events(s.env, human, 0 if full_log else self.log_sent)
        if not full_log and not peek:
            self.log_sent = len(s.env.history)
        result = None
        if self.finished and s.result is not None:
            winner = "you" if s.result.winner == human else "bot"
            result = {"winner": winner, "reason": self.end_reason or "win"}
        return {"t": "state", "view": view, "prompt": prompt,
                "log": log,
                "thinking": {"seconds": thinking} if thinking is not None else None,
                "result": result}

    async def send_state(self, *, full_log: bool = False, peek: bool = False) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send_json(self.state_message(full_log=full_log, peek=peek))
        except (WebSocketDisconnect, RuntimeError):
            return
        if not peek and self.session is not None:
            self._last_sig = self._sig()

    # --- duel flow ---

    async def advance(self) -> None:
        """Pump bot moves/singletons until a human prompt waits or the duel ends.

        A state goes out after every change (one per settle run and per bot
        move); the prompt state is not re-sent when nothing changed since the
        last send, so the client never gets the same prompt twice.
        """
        from ygo_zero.play.session import DriverDiverged

        gen = self.gen
        stop = asyncio.Event()
        ticker = asyncio.create_task(self._think_loop(stop))
        try:
            while True:
                s = self.session
                if s is None or self.ws is None or gen != self.gen:
                    return
                if s.done:
                    await self._finish(s, "")
                    return
                if s.settle("all"):
                    await self.send_state()
                if gen != self.gen:
                    return
                if s.done:
                    await self._finish(s, "")
                    return
                if s.need_human:
                    if self._sig() != self._last_sig:
                        await self.send_state()
                    return
                try:
                    await asyncio.to_thread(s.step_bot)
                except DriverDiverged as e:
                    print(f"ALERT web: product driver diverged: {e}", flush=True)
                    res = s.surrender(by_human=True)
                    res.diverged = True
                    s.write_record(res)
                    self.finished = True
                    self.end_reason = "the bot diverged (see server log)"
                    await self.send_state(full_log=True)
                    return
                if gen != self.gen:
                    return
                await self.send_state()
        finally:
            stop.set()
            await ticker

    def _sig(self) -> tuple:
        s = self.session
        return (len(s.env.history), s.env.duel.turn, s.env.duel.phase, s.done)

    async def _think_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
                return
            except TimeoutError:
                pass
            if self.thinking_seconds() is not None:
                await self.send_state(peek=True)

    async def _finish(self, s, reason: str) -> None:
        from ygo_zero.play.session import DriverDiverged

        if self.finished:
            await self.send_state(full_log=True)
            return
        try:
            await asyncio.to_thread(s.finish)
        except DriverDiverged as e:
            print(f"ALERT web: product driver diverged at finish: {e}", flush=True)
            res = s.surrender(by_human=True)
            res.diverged = True
            s.write_record(res)
            self.finished = True
            self.end_reason = "the bot diverged (see server log)"
            await self.send_state(full_log=True)
            return
        s.write_record(s.result)
        self.finished = True
        self.end_reason = reason
        self.log_sent = len(s.env.history)
        await self.send_state(full_log=True)

    # --- messages ---

    async def on_new_game(self, msg: dict) -> None:
        from ygo_zero.engine.duel import DuelConfig
        from ygo_zero.play.session import DuelSession

        deck_spec = msg.get("deck")
        bot_spec = msg.get("bot_deck", deck_spec)
        first = msg.get("first", "rps")
        mode = msg.get("mode", self.opponent_default)
        if isinstance(deck_spec, dict) and "ydk" in deck_spec:
            deck_id = f"pasted-{time.strftime('%Y%m%d-%H%M%S')}"
            self.user_decks[deck_id] = deck_spec["ydk"]
        deck, err = self.resolve_deck(deck_spec)
        if err is not None:
            await self._error(err)
            return
        bot_deck, berr = self.resolve_deck(bot_spec)
        if berr is not None:
            await self._error(berr)
            return
        if mode not in ("product", "fast", "random"):
            await self._error(f"unknown mode {mode!r} (product|fast|random)")
            return
        if first == "me":
            human_seat, first_deck, second_deck = 0, deck, bot_deck
        elif first == "bot":
            human_seat, first_deck, second_deck = 1, bot_deck, deck
        elif first == "rps":
            human_seat = random.Random().randint(0, 1)
            first_deck, second_deck = (deck, bot_deck) if human_seat == 0 else (bot_deck, deck)
        else:
            await self._error(f"unknown first {first!r} (rps|me|bot)")
            return
        seed = random.Random().getrandbits(63)
        cfg = DuelConfig(_DeckLike(first_deck), _DeckLike(second_deck), seed=seed)
        try:
            opp = await asyncio.to_thread(self._make_opponent, mode)
        except Exception as e:  # noqa: BLE001 -- report, keep the old game (if any)
            await self._error(f"could not start the {mode} opponent: {e}")
            return
        launcher = opp if hasattr(opp, "start_duel") else None
        bot = None if launcher is not None else opp
        session = DuelSession(self.engine, cfg, human_seat, bot=bot, launcher=launcher,
                              seed=seed, record_root=self.record_root, human_auto=True)
        try:
            await asyncio.to_thread(session.reset)
        except Exception as e:  # noqa: BLE001
            session.close()
            await self._error(f"could not start the duel: {e}")
            return
        # NOTE: the caller (ws_handler) already holds `srv.lock`; asyncio locks
        # are not reentrant, so this must not acquire it again.
        if self.session is not None:
            self.session.close()
        self.session = session
        self.mode = mode
        self.prompt_id = 0
        self.log_sent = 0
        self.finished = False
        self.end_reason = ""
        self._last_sig = ()
        self.gen += 1
        await self.advance()

    async def on_choose(self, msg: dict) -> None:
        s = self.session
        if s is None or (s.done and self.finished):
            await self._error("no active game (send new_game first)")
            return
        if not s.need_human:
            await self._error("no prompt is waiting for you right now")
            return
        if msg.get("prompt") != self.prompt_id:
            await self._error("stale prompt (a newer one is waiting)")
            return
        try:
            index = int(msg.get("index"))
        except (TypeError, ValueError):
            await self._error("index must be an option number")
            return
        try:
            s.step_human(index)
        except (IndexError, ValueError) as e:
            await self._error(f"illegal option: {e}")
            return
        await self.advance()

    async def on_surrender(self) -> None:
        s = self.session
        if s is None or (s.done and self.finished):
            await self._error("no active game (send new_game first)")
            return
        s.surrender(by_human=True)
        s.write_record(s.result)
        self.finished = True
        self.end_reason = "surrender"
        await self.send_state(full_log=True)

    async def _error(self, text: str) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send_json({"t": "error", "text": text})
        except (WebSocketDisconnect, RuntimeError):
            pass


class _DeckLike:
    def __init__(self, d) -> None:
        self.main = list(d.main)
        self.extra = list(d.extra)


def _card_json(code: int) -> dict:
    from ygo_zero.engine.resources import load_cards
    from ygo_zero.web.view import DB

    names = DB.names()
    text = DB.effect_text(code)
    cards = load_cards()
    cd = cards.get(code)
    st = DB.stats().get(code, (0, 0, 0, 0, 0, 0))
    atk, dfn, level, rank, link, type_ = st
    if cd is not None:
        atk, dfn = cd.attack, cd.defense
    return {"code": code, "name": names.get(code, f"#{code}"),
            "type": type_, "type_str": _type_str(type_),
            "attribute": cd.attribute if cd is not None else 0,
            "race": cd.race if cd is not None else 0,
            "level": level, "rank": rank, "link": link,
            "atk": atk, "def": dfn, "text": text}


def _type_str(type_: int) -> str:
    from ygo_zero.engine import constants as C

    if type_ & C.TYPE_MONSTER:
        kinds = []
        for bit, name in ((C.TYPE_FUSION, "Fusion"), (C.TYPE_SYNCHRO, "Synchro"),
                          (C.TYPE_XYZ, "Xyz"), (C.TYPE_LINK, "Link"),
                          (C.TYPE_RITUAL, "Ritual"), (C.TYPE_PENDULUM, "Pendulum"),
                          (C.TYPE_TUNER, "Tuner"), (C.TYPE_EFFECT, "Effect"),
                          (C.TYPE_NORMAL, "Normal")):
            if type_ & bit:
                kinds.append(name)
        return " ".join(kinds) + " Monster" if kinds else "Monster"
    if type_ & C.TYPE_SPELL:
        return "Spell"
    if type_ & C.TYPE_TRAP:
        return "Trap"
    return f"type {type_:#x}"


def create_app(*, token: str, engine=None, make_opponent=None,
               record_root: Path = ROOT / "runs" / "web",
               pics_dir: Path = PICS_DIR, opponent_default: str = "product",
               device: str = "cpu", replay_threads: int = 1) -> Starlette:
    srv = Server(token=token, engine=engine, make_opponent=make_opponent,
                 record_root=record_root, pics_dir=pics_dir,
                 opponent_default=opponent_default, device=device,
                 replay_threads=replay_threads)

    def need_token(req) -> bool:
        return req.query_params.get("t") == srv.token

    async def index(req):
        if not need_token(req):
            return Response("unauthorized", status_code=401)
        page = ROOT / "web" / "index.html"
        if page.is_file():
            return Response(page.read_text(encoding="utf-8"),
                            media_type="text/html")
        return Response(PLACEHOLDER, media_type="text/html")

    async def card(req):
        if not need_token(req):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            code = int(req.path_params["code"])
        except ValueError:
            return JSONResponse({"error": "bad code"}, status_code=404)
        return JSONResponse(_card_json(code))

    async def img(req):
        if not need_token(req):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            code = int(req.path_params["code"])
        except ValueError:
            return JSONResponse({"error": "bad code"}, status_code=404)
        size = req.query_params.get("size", "small")
        if size not in ("small", "full"):
            return JSONResponse({"error": "size must be small|full"}, status_code=400)
        if code == 0:
            return Response(CARD_BACK_SVG, media_type="image/svg+xml")
        cached = srv.pics_dir / size / f"{code}.jpg"
        if cached.is_file():
            return Response(cached.read_bytes(), media_type="image/jpeg")
        try:
            data = await asyncio.to_thread(
                _fetch_img, IMG_URLS[size].format(code=code))
        except Exception:  # noqa: BLE001 -- offline: the card-back placeholder
            return Response(CARD_BACK_SVG, media_type="image/svg+xml")
        if data[:2] != b"\xff\xd8":
            return Response(CARD_BACK_SVG, media_type="image/svg+xml")
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
        return Response(data, media_type="image/jpeg")

    async def ws_handler(ws: WebSocket):
        if ws.query_params.get("t") != srv.token:
            await ws.close(code=4401)
            return
        await ws.accept()
        old = srv.ws
        srv.ws = ws
        if old is not None and old is not ws:
            try:
                await old.close(code=4400)
            except (WebSocketDisconnect, RuntimeError):
                pass
        try:
            await ws.send_json({"t": "lobby", "decks": srv.list_decks()})
            if srv.session is not None and not (srv.session.done and srv.finished):
                await srv.send_state(full_log=True)
            while True:
                try:
                    raw = await ws.receive_text()
                except WebSocketDisconnect:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await srv._error("not JSON")
                    continue
                if srv.ws is not ws:
                    break  # taken over by a newer connection
                async with srv.lock:
                    t = msg.get("t")
                    if t == "new_game":
                        await srv.on_new_game(msg)
                    elif t == "choose":
                        await srv.on_choose(msg)
                    elif t == "surrender":
                        await srv.on_surrender()
                    elif t == "resync":
                        if srv.session is None:
                            await ws.send_json({"t": "lobby",
                                                "decks": srv.list_decks()})
                        else:
                            await srv.send_state(full_log=True)
                    else:
                        await srv._error(f"unknown message {t!r}")
        finally:
            if srv.ws is ws:
                srv.ws = None

    routes = [Route("/", index), Route("/card/{code}", card),
              Route("/img/{code}.jpg", img), WebSocketRoute("/ws", ws_handler)]
    web_dir = ROOT / "web"
    if web_dir.is_dir():  # client code is public; game data (ws, img, card) needs the token
        routes.append(Mount("/", StaticFiles(directory=web_dir)))
    app = Starlette(routes=routes)
    app.state.server = srv
    return app
