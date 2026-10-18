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
        self.cue_sent = 0
        self.gen = 0
        self.finished = False
        self.end_reason = ""
        self._last_sig: tuple = ()
        self.user_decks: dict[str, str] = {}  # id -> ydk text (pasted this session)
        self.chain_mode = "auto"  # "auto": skip quiet chain windows (session.auto_pass_index); "all"
        self.auto_passed: set[int] = set()  # history numbers (1-based n) of auto-passes
        self.stats = None  # DuelStats | None (one per game, viewer = human seat)
        self._adv: asyncio.Task | None = None
        self.advancing = False

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
        from ygo_zero.web.cues import build_cues
        from ygo_zero.web.stats import DuelStats
        from ygo_zero.web.view import build_events, build_prompt, build_view

        s = self.session
        if s is None:
            return {"t": "lobby", "decks": self.list_decks()}
        human = s.human_seat
        if self.stats is None or getattr(self.stats, "viewer", human) != human:
            self.stats = DuelStats(human)
        view = build_view(s.env, human)
        prompt = None
        if s.need_human and not s.done and not self.advancing:  # not one auto-pass may skip
            self.prompt_id = s.current_prompt_id()
            prompt = build_prompt(s.env, human, self.prompt_id)
        thinking = self.thinking_seconds()
        log = build_events(s.env, human, 0 if full_log else self.log_sent)
        for e in log:
            if e["n"] in self.auto_passed:
                e["actor"], e["text"], e["card"], e["kind"] = "system", "(auto-passed)", None, "auto_pass"
        if not peek:
            cues, self.cue_sent = build_cues(s.env, human, since=self.cue_sent)
            self.stats.feed_cues(cues)
        else:
            cues = []  # thinking ticker: transient cues wait for the real send
        self.stats.note_view(view)
        if not full_log and not peek:
            self.log_sent = len(s.env.history)
        result = None
        if self.finished and s.result is not None:
            winner = "you" if s.result.winner == human else "bot"
            result = {"winner": winner, "reason": self.end_reason or "win"}
        return {"t": "state", "view": view, "prompt": prompt,
                "log": log, "cues": cues, "stats": self.stats.stats_message(),
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
                moved = s.settle("all")
                if s.done:
                    await self._finish(s, "")
                    return
                if s.need_human and self.chain_mode == "auto":
                    i = s.auto_pass_index()
                    if i is not None:  # never shown: the client must not answer it
                        try:
                            had = any(o.kind.name == "ACTIVATE" for o in s.env.opts)
                            s.step_human(i)
                        except DriverDiverged as e:
                            await self._diverged(s, e, gen)
                            return
                        if self.stats is not None:
                            self.stats.note_chain_window(had, False)
                        self.auto_passed.add(len(s.env.history))
                        continue
                if s.need_human:
                    self.advancing = False  # the client may answer as soon as it sees this
                    if self._sig() != self._last_sig:
                        await self.send_state()
                    return
                if moved:
                    await self.send_state()
                if gen != self.gen:
                    return
                try:
                    await asyncio.to_thread(s.step_bot)
                except DriverDiverged as e:
                    await self._diverged(s, e, gen)
                    return
                if gen != self.gen:
                    return
                await self.send_state()
        finally:
            if gen == self.gen:
                self.advancing = False
            stop.set()
            await ticker

    async def _diverged(self, s, e, gen: int) -> None:
        if gen != self.gen:  # abandoned on purpose (surrender / new game): not a divergence
            return
        print(f"ALERT web: product driver diverged: {e}", flush=True)
        res = s.surrender(by_human=True)
        res.diverged = True
        s.write_record(res)
        self.finished = True
        self.end_reason = "the bot diverged (see server log)"
        await self.send_state(full_log=True)

    def start_advance(self) -> None:
        """Run `advance` as a task, so the socket keeps reading (surrender must cut
        through a thinking bot)."""
        self.advancing = True
        self._adv = asyncio.create_task(self.advance())
        self._adv.add_done_callback(self._advance_done)

    def _advance_done(self, task: asyncio.Task) -> None:
        """A crash in the background advance must surface, never leave the client
        waiting on a prompt that will not come (fail fast and loud)."""
        if task.cancelled() or task.exception() is None:
            return
        import traceback

        e = task.exception()
        print("ALERT web: advance crashed:\n" + "".join(traceback.format_exception(e)), flush=True)
        self.advancing = False
        asyncio.get_running_loop().create_task(self._error(f"server error: {e!r} (see server log)"))

    def _drop_session(self, s) -> None:
        """Stop a session that may still have a bot step in flight; close it off-loop."""
        s.abandon()
        task = asyncio.create_task(asyncio.to_thread(s.close))
        task.add_done_callback(lambda t: t.exception())

    def _sig(self) -> tuple:
        s = self.session
        return (len(s.env.history), s.env.duel.turn, s.env.duel.phase, s.done,
                s.need_human and not self.advancing)

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
        chain = msg.get("chain", self.chain_mode)
        if chain in ("auto", "all"):
            self.chain_mode = chain
        meta = {"human_deck": deck_spec if isinstance(deck_spec, str) else "pasted",
                "bot_deck": bot_spec if isinstance(bot_spec, str) else "pasted",
                "mode": mode, "first": first, "chain": self.chain_mode}
        session = DuelSession(self.engine, cfg, human_seat, bot=bot, launcher=launcher,
                              seed=seed, record_root=self.record_root, human_auto=True,
                              record_meta=meta)
        try:
            await asyncio.to_thread(session.reset)
        except Exception as e:  # noqa: BLE001
            session.close()
            await self._error(f"could not start the duel: {e}")
            return
        # NOTE: the caller (ws_handler) already holds `srv.lock`; asyncio locks
        # are not reentrant, so this must not acquire it again.
        self.gen += 1
        if self.session is not None:
            self._drop_session(self.session)
        self.session = session
        self.mode = mode
        self.prompt_id = 0
        self.log_sent = 0
        self.cue_sent = 0
        self.finished = False
        self.end_reason = ""
        self._last_sig = ()
        self.auto_passed = set()
        from ygo_zero.web.stats import DuelStats

        self.stats = DuelStats(session.human_seat)
        self.start_advance()

    async def on_choose(self, msg: dict) -> None:
        s = self.session
        if s is None or (s.done and self.finished):
            await self._error("no active game (send new_game first)")
            return
        if not s.need_human or self.advancing:
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
            from ygo_zero.engine import messages as M

            dec = s.env.pending_decision()
            is_chain = dec is not None and isinstance(dec.msg, M.SelectChain) \
                and s.env.player == s.human_seat
            had = bool(is_chain) and any(o.kind.name == "ACTIVATE" for o in s.env.opts)
            used = bool(is_chain and 0 <= index < len(s.env.opts)
                        and s.env.opts[index].kind.name == "ACTIVATE")
            s.step_human(index)
        except (IndexError, ValueError) as e:
            await self._error(f"illegal option: {e}")
            return
        if self.stats is not None and had:
            self.stats.note_chain_window(True, used)
        self.start_advance()

    async def on_surrender(self) -> None:
        s = self.session
        if s is None or (s.done and self.finished):
            await self._error("no active game (send new_game first)")
            return
        self.gen += 1  # stops a running advance; a bot step in flight is abandoned
        thinking = s.thinking_since is not None
        s.surrender(by_human=True)
        s.write_record(s.result)
        self.finished = True
        self.end_reason = "surrender"
        if thinking:
            s.abandon()
        await self.send_state(full_log=True)

    async def on_settings(self, msg: dict) -> None:
        chain = msg.get("chain")
        if chain not in ("auto", "all"):
            await self._error("settings.chain must be auto|all")
            return
        self.chain_mode = chain
        s = self.session
        if chain == "auto" and s is not None and not self.finished and s.need_human \
                and not self.advancing:
            self.start_advance()  # a quiet window may be waiting right now

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


def _register_stack_dump() -> None:
    """`kill -USR1 <server pid>` prints every thread's stack (stall diagnosis)."""
    import faulthandler
    import signal

    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except (AttributeError, ValueError):  # not on this platform / not the main thread
        pass


def create_app(*, token: str, engine=None, make_opponent=None,
               record_root: Path = ROOT / "runs" / "web",
               pics_dir: Path = PICS_DIR, opponent_default: str = "product",
               device: str = "cpu", replay_threads: int = 1) -> Starlette:
    _register_stack_dump()
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
        from ygo_zero.engine.resources import extra_card_dirs

        for d in extra_card_dirs():
            cand = d / "pics" / f"{code}.jpg"
            if cand.is_file():
                return Response(cand.read_bytes(), media_type="image/jpeg")
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
                except (WebSocketDisconnect, RuntimeError):
                    break  # closed, or taken over (starlette: 'Need to call "accept" first')
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await srv._error("not JSON")
                    continue
                if srv.ws is not ws:
                    break  # taken over by a newer connection
                t = msg.get("t") if isinstance(msg, dict) else None
                if t == "surrender":  # never waits behind the bot
                    await srv.on_surrender()
                    continue
                if t == "settings":
                    await srv.on_settings(msg)
                    continue
                async with srv.lock:
                    if t == "new_game":
                        await srv.on_new_game(msg)
                    elif t == "choose":
                        await srv.on_choose(msg)
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
