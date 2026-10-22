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
PUZZLES_PATH = ROOT / "src" / "decklight" / "web" / "puzzles.json"
PRIMERS_PATH = ROOT / "src" / "decklight" / "web" / "primers.json.gz"  # scripts/build_primers.py --commit-copy
STUDIED_DIR = ROOT / "data" / "primers" / "studied"
STUDIED_PRIMERS_PATH = STUDIED_DIR / "primers.json"
STUDY_PROCS = 3
STUDY_GAMES = 32

IMG_URLS = {
    "small": "https://images.ygoprodeck.com/images/cards_small/{code}.jpg",
    "full": "https://images.ygoprodeck.com/images/cards/{code}.jpg",
}

PLACEHOLDER = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>decklight web client</title></head><body>
<h1>decklight web client</h1>
<p>The phone client (W2) is not built yet. The game protocol lives at <code>/ws</code>.</p>
</body></html>
"""

CARD_BACK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="146" viewBox="0 0 100 146">
<rect width="100" height="146" rx="6" fill="#1d2a5e"/>
<rect x="8" y="8" width="84" height="130" rx="4" fill="none" stroke="#c8a24a" stroke-width="2"/>
<circle cx="50" cy="73" r="22" fill="none" stroke="#c8a24a" stroke-width="2"/>
<text x="50" y="77" text-anchor="middle" font-size="10" fill="#c8a24a">decklight</text>
</svg>
"""


def _fetch_img(url: str, timeout: float = 5.0) -> bytes:
    """One image fetch (module-level so tests can monkeypatch it)."""
    req = urllib.request.Request(url, headers={"User-Agent": "decklight-web/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _pretty_deck_name(stem: str) -> str:
    return re.sub(r"-meta\d+$", "", stem).replace("-", " ").title()


class Server:
    """App state: decks, the engine, the opponent factory, and the live game."""

    def __init__(
        self,
        *,
        token: str,
        engine=None,
        make_opponent=None,
        record_root: Path = ROOT / "runs" / "web",
        pics_dir: Path = PICS_DIR,
        opponent_default: str = "product",
        device: str = "cpu",
        replay_threads: int = 1,
    ) -> None:
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
        # The thinking ticker resends this snapshot (taken on the event loop before the bot thread starts): querying the
        # live duel while step_bot applies a move races on the core's query buffer (struct.error, fresh-clone-20261021-2108).
        self._peek_msg: dict | None = None
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
        self.winchance_on = False  # client setting, off by default, reset per connection
        self.winchance = None  # {"p","turn","history"} while on (sent as null when off)
        self._wc_n = 0  # winchance computations this game (forces the prompt state out)
        self._wc_map = None  # lazy CalibMap (False = unavailable, ALERTed)
        self._wc_ms = 0.0  # last value-forward wall ms (CPU)
        self.auto_passed: set[int] = set()  # history numbers (1-based n) of auto-passes
        self.practice = False  # practice mode: hints allowed, excluded from human stats
        self.hints_used = 0  # hints asked this game (recorded in the record)
        self.undos = 0  # practice undos this game (recorded in the record)
        # Host choice index of each real (multi-option) human decision this game,
        # in order; undo rewinds to the last one (auto-passed windows are not here).
        self.human_decisions: list[int] = []
        self.what_if: dict | None = None  # {"source","k","option","turn"} for a what-if game
        self.puzzle: dict | None = None  # {"id","start_turn","goal","status"} for a puzzle game
        self._puzzles_cache: list[dict] | None = None
        self._primers_cache: dict | None = None
        # Decks of the live game, kept for the in-game opponent primer.
        self.game_decks: dict | None = None  # {"human": Deck, "bot": Deck}
        self.game_deck_names: dict = {}  # {"human": str, "bot": str}
        self.record_path: Path | None = None  # game.jsonl of the live/finished game
        self._review_task: asyncio.Task | None = None
        self.stats = None  # DuelStats | None (one per game, viewer = human seat)
        self._adv: asyncio.Task | None = None
        self.advancing = False
        self._study_proc: asyncio.subprocess.Process | None = None  # at most one study at a time

    # --- lazy resources ---

    @property
    def engine(self):
        if self._engine is None:
            from decklight.engine.duel import Engine

            self._engine = Engine()
        return self._engine

    def emb_codes(self) -> set[int]:
        if self._emb_codes is None:
            import numpy as np

            from decklight.edopro.host import EMB_PATH

            self._emb_codes = {int(c) for c in np.load(EMB_PATH)["codes"]}
        return self._emb_codes

    def _default_opponent(self, mode: str):
        from decklight.edopro.bot import RandomBot
        from decklight.edopro.product_bot import ProductLauncher

        if mode == "random":
            return RandomBot(random.Random().randrange(1 << 30))
        if mode == "fast":
            return ProductLauncher(device=self.device, fast=True, replay_threads=self.replay_threads)
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
                out.append({"id": p.stem, "name": _pretty_deck_name(p.stem), "group": group})
        for i, (deck_id, _text) in enumerate(sorted(self.user_decks.items())):
            out.append({"id": deck_id, "name": f"Pasted deck {i + 1}", "group": "user"})
        return out

    def resolve_deck(self, spec):
        """A deck id or {"ydk": text} -> (Deck, error text|None)."""
        from decklight.decks import load_ydk, parse_ydk

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
        return None, 'deck must be a deck id or {"ydk": "..."}'

    def check_text(self, deck) -> str | None:
        """The EDOPro host's deck rules; names cards without embeddings."""
        from decklight.demos.names import CardNames
        from decklight.edopro.host import check_deck, split_deck

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
        return f"illegal deck ({what}: have {cur}, allowed {minimum}-{maximum}){extra_txt}"

    # --- state ---

    def thinking_seconds(self) -> float | None:
        s = self.session
        if s is not None and s.thinking_since is not None and not s.done:
            return time.monotonic() - s.thinking_since
        return None

    def state_message(self, *, full_log: bool = False, peek: bool = False) -> dict:
        from decklight.web.cues import build_cues
        from decklight.web.stats import DuelStats
        from decklight.web.view import build_events, build_prompt, build_view

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
        return {
            "t": "state",
            "view": view,
            "prompt": prompt,
            "log": log,
            "cues": cues,
            "stats": self.stats.stats_message(winchance_on=self.winchance_on),
            "winchance": self.winchance if self.winchance_on else None,
            "thinking": {"seconds": thinking} if thinking is not None else None,
            "result": result,
            "practice": self.practice,
            "can_undo": bool(self.practice and not self.advancing and s.need_human and self.human_decisions),
            "what_if": self.what_if,
            "puzzle": self.puzzle,
        }

    def get_puzzles(self) -> list[dict]:
        """Puzzles from puzzles.json, loaded once; missing file -> empty list."""
        if self._puzzles_cache is not None:
            return self._puzzles_cache
        try:
            raw = PUZZLES_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            out = data if isinstance(data, list) else []
        except (OSError, ValueError):
            out = []
        self._puzzles_cache = out if isinstance(out, list) else []
        return self._puzzles_cache

    def _puzzle_item(self, p: dict) -> dict:
        decks = [str(d) for d in p.get("decks", [])]
        names = [_pretty_deck_name(d) for d in decks[:2]]
        while len(names) < 2:
            names.append("?")
        return {
            "id": str(p.get("id")),
            "decks": names,
            "turn": int(p.get("turn", 1)),
            "difficulty": str(p.get("difficulty", "medium")),
            "goal": str(p.get("goal", "")),
            "n_plays": int(p.get("n_plays", p.get("n_moves", 0))),
        }

    def get_primers(self) -> dict:
        """Opponent primers from primers.json, merged with studied lists; missing file -> empty store.

        Studied entries (data/primers/studied/primers.json) never replace a
        shipped entry with the same hash.
        """
        if self._primers_cache is not None:
            return self._primers_cache
        from decklight.analysis.primer import load_store

        try:
            data = load_store(PRIMERS_PATH)
            out = data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            out = {}
        if not isinstance(out.get("decks"), dict):
            out = {"decks": {}, "negated_by": {}}
        out.setdefault("negated_by", {})
        try:
            studied = json.loads(STUDIED_PRIMERS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            studied = {}
        if isinstance(studied.get("decks"), dict):
            for h, e in studied["decks"].items():
                if h not in out["decks"]:
                    out["decks"][h] = e
            sneg = studied.get("negated_by", {})
            if isinstance(sneg, dict):
                neg = out["negated_by"]
                for k, victims in sneg.items():
                    if not isinstance(victims, dict):
                        continue
                    slot = neg.setdefault(k, {})
                    for v, n in victims.items():
                        slot[v] = slot.get(v, 0) + n
        self._primers_cache = out
        return self._primers_cache

    @staticmethod
    def _trim_primer(p: dict | None) -> dict | None:
        """What the client sheet shows (no files, no bot-vs-bot win rates)."""
        if p is None:
            return None
        out = {
            "archetype": p.get("archetype", ""),
            "games": int(p.get("games", 0)),
            "key_cards": [
                {
                    "code": c.get("code"),
                    "name": c.get("name", ""),
                    "games": int(c.get("games", 0)),
                    "per_game": c.get("per_game", 0.0),
                    **({k: c[k] for k in ("impact", "per_play") if k in c}),
                }
                for c in p.get("key_cards", [])[:8]
            ],
            "first_plays": [
                {
                    "code": c.get("code"),
                    "name": c.get("name", ""),
                    "games": int(c.get("games", 0)),
                    "share": c.get("share", 0.0),
                }
                for c in p.get("first_plays", [])[:4]
            ],
            "first_n": int(p.get("first_n", 0)),
            # No chokepoints and no answer gains: neither ranking validated on held-out games (user chose to hide
            # "Where it gets stopped", NOTEBOOK 2026-10-18 primer comparison).
            "your_answers": [
                {
                    "code": a.get("code"),
                    "name": a.get("name", ""),
                    "hits": [
                        {
                            "code": h.get("code"),
                            "name": h.get("name", ""),
                            "times": int(h.get("times", 0)),
                        }
                        for h in a.get("hits", [])
                    ],
                    **({"note": a["note"]} if a.get("note") else {}),
                }
                for a in p.get("your_answers", [])
            ],
        }
        if p.get("similar") is not None:
            s = p["similar"]
            out["similar"] = {
                "archetype": s.get("archetype", ""),
                "match": int(s.get("match", 0)),
                "of": int(s.get("of", 0)),
            }
        if p.get("study") is not None:
            st = p["study"]
            out["study"] = {"phase": st.get("phase", ""), "games": int(st.get("games", 0))}
        return out

    def _remember_decks(self, human_deck, bot_deck, human_spec=None, bot_spec=None) -> None:
        """Keep the live game's Deck objects for the in-game opponent primer."""

        def name(spec):
            return _pretty_deck_name(spec) if isinstance(spec, str) and spec != "pasted" else "Pasted deck"

        self.game_decks = {"human": human_deck, "bot": bot_deck}
        self.game_deck_names = {"human": name(human_spec), "bot": name(bot_spec)}

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
        from decklight.play.session import DriverDiverged

        gen = self.gen
        stop = asyncio.Event()
        ticker = asyncio.create_task(self._think_loop(stop))
        try:
            while True:
                s = self.session
                if s is None or self.ws is None or gen != self.gen:
                    return
                if self._puzzle_failed(s):
                    await self._fail_puzzle(s)
                    return
                if s.done:
                    await self._finish(s, "")
                    return
                moved = s.settle("all")
                if self._puzzle_failed(s):
                    await self._fail_puzzle(s)
                    return
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
                    if self.winchance_on:
                        self._compute_winchance()
                    if self._sig() != self._last_sig:
                        await self.send_state()
                    return
                if moved:
                    await self.send_state()
                if gen != self.gen:
                    return
                self._peek_msg = self.state_message(peek=True)
                try:
                    await asyncio.to_thread(s.step_bot)
                except DriverDiverged as e:
                    await self._diverged(s, e, gen)
                    return
                finally:
                    self._peek_msg = None
                if gen != self.gen:
                    return
                if self._puzzle_failed(s):
                    await self._fail_puzzle(s)
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
        return (
            len(s.env.history),
            s.env.duel.turn,
            s.env.duel.phase,
            s.done,
            s.need_human and not self.advancing,
            self._wc_n,
        )

    def _product_value_fn(self):
        """The product's value net, or None (random/fast mode: no value net)."""
        s = self.session
        if s is None or getattr(s, "launcher", None) is None:
            return None
        return getattr(s.launcher, "value", None)

    def _compute_winchance(self) -> None:
        """Value_p2 on the human's own pending observation, calibrated.

        Call only while a human prompt waits (env.player == the human seat), so
        the number can never leak the bot's hidden hand through a bot-view
        observation. Between human decisions the last value is kept.
        """
        import numpy as np

        s = self.session
        fn = self._product_value_fn()
        if s is None or fn is None or not s.need_human:
            return
        if self._wc_map is None:
            try:
                from decklight.eval.winchance import load_calib

                self._wc_map = load_calib()
            except Exception as e:  # noqa: BLE001 -- map missing: stay null, keep playing
                print(f"ALERT web: winchance map unavailable: {e}", flush=True)
                self._wc_map = False
                return
        if self._wc_map is False:
            return
        t0 = time.monotonic()
        # The same builder step() returns: the pending decision's observation,
        # always from the player to act's view (here: the human seat).
        obs = s.env._observe()
        batch = {k: np.ascontiguousarray(v)[np.newaxis, ...] for k, v in obs.items()}
        v = float(np.asarray(fn(batch)).ravel()[0])
        self._wc_ms = (time.monotonic() - t0) * 1000.0
        turn = int(s.env.duel.turn)
        p = self._wc_map.p(v, turn)
        self._wc_n += 1
        if self.winchance is None:
            self.winchance = {"p": p, "turn": turn, "history": []}
        else:
            self.winchance["p"] = p
            self.winchance["turn"] = turn
        self.winchance["history"].append([turn, p])
        if self.stats is not None:
            self.stats.note_winchance(turn, p)

    async def _think_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
                return
            except TimeoutError:
                pass
            await self._send_peek()

    async def _send_peek(self) -> None:
        """The thinking ticker: the pre-move snapshot with the current `thinking.seconds` (never touches the duel)."""
        msg, t = self._peek_msg, self.thinking_seconds()
        if msg is None or t is None or self.ws is None:
            return
        try:
            await self.ws.send_json({**msg, "thinking": {"seconds": t}})
        except (WebSocketDisconnect, RuntimeError):
            return

    async def _finish(self, s, reason: str) -> None:
        from decklight.play.session import DriverDiverged

        if self.finished:
            await self.send_state(full_log=True)
            return
        try:
            await asyncio.to_thread(s.finish)
        except DriverDiverged as e:
            print(f"ALERT web: product driver diverged at finish: {e}", flush=True)
            res = s.surrender(by_human=True)
            res.diverged = True
            s.record_meta["hints_used"] = self.hints_used
            if self.puzzle is not None:
                s.record_meta["puzzle_result"] = "failed"
                self.puzzle = dict(self.puzzle, status="failed")
            self.record_path = s.write_record(res, auto_passed=self.auto_passed)
            self.finished = True
            self.end_reason = "the bot diverged (see server log)"
            await self.send_state(full_log=True)
            return
        s.record_meta["hints_used"] = self.hints_used
        if self.puzzle is not None and s.result is not None:
            start = int(self.puzzle.get("start_turn", 0))
            seat = int(s.human_seat)
            won = s.result.winner == seat and int(s.env.duel.turn) == start
            status = "solved" if won else "failed"
            self.puzzle = dict(self.puzzle, status=status)
            s.record_meta["puzzle_result"] = status
        self.record_path = s.write_record(s.result, auto_passed=self.auto_passed)
        self.finished = True
        self.end_reason = reason
        self.log_sent = len(s.env.history)
        await self.send_state(full_log=True)

    def _puzzle_failed(self, s) -> bool:
        """A puzzle fails the moment the turn passes without a win (game stops there)."""
        if self.puzzle is None or s is None or s.done or self.finished:
            return False
        try:
            start = int(self.puzzle.get("start_turn", 0))
        except (TypeError, ValueError):
            return False
        try:
            cur = int(s.env.duel.turn)
        except Exception:  # noqa: BLE001 -- no board yet: not failed
            return False
        return cur != start

    async def _fail_puzzle(self, s) -> None:
        """Stop a failed puzzle: abandon, write the record with puzzle_result, send failed."""
        from decklight.play.session import DuelResult

        if self.finished:
            return
        thinking = s.thinking_since is not None
        try:
            res = DuelResult(
                None, int(s.env.duel.turn), list(s.env.choices), [e.actor for e in s.env.history]
            )
        except Exception:  # noqa: BLE001 -- best effort record
            res = s.surrender(by_human=True)
        s.record_meta["hints_used"] = self.hints_used
        s.record_meta["puzzle_result"] = "failed"
        if self.puzzle is not None:
            self.puzzle = dict(self.puzzle, status="failed")
        self.record_path = s.write_record(res, auto_passed=self.auto_passed)
        self.finished = True
        self.end_reason = "puzzle failed"
        if thinking:
            s.abandon()
        self.log_sent = len(s.env.history)
        await self.send_state(full_log=True)

    # --- messages ---

    async def on_new_game(self, msg: dict) -> None:
        from decklight.engine.duel import DuelConfig
        from decklight.play.session import DuelSession

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
        practice = msg.get("practice", False)
        if not isinstance(practice, bool):
            await self._error("practice must be true|false")
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
        meta = {
            "human_deck": deck_spec if isinstance(deck_spec, str) else "pasted",
            "bot_deck": bot_spec if isinstance(bot_spec, str) else "pasted",
            "mode": mode,
            "first": first,
            "chain": self.chain_mode,
            "practice": practice,
            "hints_used": 0,
        }
        session = DuelSession(
            self.engine,
            cfg,
            human_seat,
            bot=bot,
            launcher=launcher,
            seed=seed,
            record_root=self.record_root,
            human_auto=True,
            record_meta=meta,
        )
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
        self._remember_decks(deck, bot_deck, deck_spec, bot_spec)
        self.practice = practice
        self.hints_used = 0
        self.undos = 0
        self.human_decisions = []
        self.what_if = None
        self.puzzle = None
        self.prompt_id = 0
        self.log_sent = 0
        self.cue_sent = 0
        self.finished = False
        self.end_reason = ""
        self._last_sig = ()
        self.winchance = None
        self._wc_n = 0
        self.auto_passed = set()
        self.record_path = None
        if self._review_task is not None and not self._review_task.done():
            self._review_task.cancel()
            self._review_task = None
        from decklight.web.stats import DuelStats

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
            from decklight.engine import messages as M

            dec = s.env.pending_decision()
            is_chain = dec is not None and isinstance(dec.msg, M.SelectChain) and s.env.player == s.human_seat
            had = bool(is_chain) and any(o.kind.name == "ACTIVATE" for o in s.env.opts)
            used = bool(
                is_chain and 0 <= index < len(s.env.opts) and s.env.opts[index].kind.name == "ACTIVATE"
            )
            before = len(s.env.choices)  # this decision's host index (for undo)
            s.step_human(index)
            self.human_decisions.append(before)
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
        s.record_meta["hints_used"] = self.hints_used
        self.record_path = s.write_record(s.result, auto_passed=self.auto_passed)
        self.finished = True
        self.end_reason = "surrender"
        if thinking:
            s.abandon()
        await self.send_state(full_log=True)

    async def on_puzzles(self) -> None:
        items = [self._puzzle_item(p) for p in self.get_puzzles()]
        if self.ws is None:
            return
        try:
            await self.ws.send_json({"t": "puzzles", "items": items})
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def on_primer(self, msg: dict) -> None:
        """The opponent deck's primer (lobby: explicit decks; in game: the live ones)."""
        from decklight.analysis.primer import hash_of, primer_for

        store = self.get_primers()
        spec = msg.get("deck")
        if spec is not None:
            bot_deck, berr = self.resolve_deck(spec)
            if berr is not None:
                await self._error(berr)
                return
            my_spec = msg.get("my_deck")
            if my_spec is not None:
                my_deck, merr = self.resolve_deck(my_spec)
                if merr is not None:
                    await self._error(merr)
                    return
            elif self.game_decks is not None:
                my_deck = self.game_decks["human"]
            else:
                await self._error("primer needs my_deck")
                return
            bot_name = _pretty_deck_name(spec) if isinstance(spec, str) else "Pasted deck"
        else:
            if self.game_decks is None:
                await self._error("no active game (send new_game first)")
                return
            bot_deck = self.game_decks["bot"]
            my_deck = self.game_decks["human"]
            bot_name = self.game_deck_names.get("bot", "Opponent")
        reply = {
            "t": "primer",
            "deck": bot_name,
            "hash": hash_of(bot_deck),
            "primer": self._trim_primer(primer_for(store, bot_deck, my_deck)),
        }
        if reply["primer"] is None:
            reply["can_study"] = True
        if self.ws is None:
            return
        try:
            await self.ws.send_json(reply)
        except (WebSocketDisconnect, RuntimeError):
            pass

    def _study_busy(self) -> bool:
        return self._study_proc is not None and self._study_proc.returncode is None

    async def shutdown(self) -> None:
        """Kill a running deck study (called on server shutdown)."""
        proc = self._study_proc
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:  # noqa: BLE001, S110 -- shutting down: best effort
                pass
        self._study_proc = None

    def _deck_text(self, spec) -> str | None:
        """The .ydk text behind a deck spec (id, pasted dict, or session id)."""
        if isinstance(spec, dict) and "ydk" in spec:
            return spec["ydk"]
        if isinstance(spec, str):
            for _group, d in DECK_DIRS:
                p = d / f"{spec}.ydk"
                if p.is_file():
                    return p.read_text(encoding="utf-8")
            if spec in self.user_decks:
                return self.user_decks[spec]
        return None

    async def on_study(self, msg: dict) -> None:
        """Study a deck list on demand: play bot games with it in the background.

        Replies ``study_status`` as the worker progresses; the primer cache is
        dropped when the first (frequency) version lands and when scoring
        finishes, so the next ``primer`` request returns the new entry.
        """
        import sys
        import tempfile

        from decklight.analysis.primer import MIN_GAMES, hash_of

        spec = msg.get("deck")
        if spec is None:
            if self.game_decks is None:
                await self._error("no active game (send new_game first)")
                return
            deck = self.game_decks["bot"]
            deck_name = self.game_deck_names.get("bot", "Opponent")
            text = "#main\n" + "\n".join(str(c) for c in deck.main)
            text += "\n#extra\n" + "\n".join(str(c) for c in deck.extra)
            text += "\n!side\n" + "\n".join(str(c) for c in getattr(deck, "side", ()))
            text += "\n"
        else:
            deck, err = self.resolve_deck(spec)
            if err is not None:
                await self._error(err)
                return
            deck_name = _pretty_deck_name(spec) if isinstance(spec, str) else "Pasted deck"
            text = self._deck_text(spec)
            if text is None:
                await self._error(f"unknown deck {spec!r}")
                return
        h = hash_of(deck)
        if self._study_busy():
            await self._send_study({"t": "study_status", "busy": True, "deck": deck_name, "hash": h})
            return
        existing = self.get_primers().get("decks", {}).get(h)
        if existing is not None and ("study" not in existing or existing["study"].get("phase") == "scored"):
            parts = []
            if existing.get("games", 0) >= MIN_GAMES:
                parts.append("shipped" if "study" not in existing else "scored")
            if parts:
                await self._error(
                    f"already studied ({parts[0]} primer with {existing.get('games', 0)} games)"
                )
                return
        with tempfile.NamedTemporaryFile("w", suffix=".ydk", delete=False, encoding="utf-8") as tmp:
            tmp.write(text)
            tmp_name = tmp.name
        try:
            self._study_proc = await asyncio.create_subprocess_exec(
                "taskset",
                "-c",
                "0-3",
                "nice",
                "-n",
                "19",
                sys.executable,
                "scripts/study_deck.py",
                "--ydk",
                tmp_name,
                "--out-dir",
                str(STUDIED_DIR),
                "--procs",
                str(STUDY_PROCS),
                "--games",
                str(STUDY_GAMES),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=ROOT,
            )
        except OSError as e:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
            await self._error(f"could not start the study: {e}")
            return
        proc = self._study_proc
        asyncio.create_task(self._watch_study(proc, tmp_name, deck_name, h))

    async def _send_study(self, msg: dict) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send_json(msg)
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def _watch_study(self, proc, ydk_tmp: str, deck_name: str, h: str) -> None:
        """Relay the study worker's STUDY lines as study_status; drop the primer
        cache when the frequency version lands and when scoring finishes."""
        import re

        total = STUDY_GAMES
        last = ""
        try:
            assert proc.stdout is not None
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                last = line
                m = re.match(r"^STUDY games (\d+)/(\d+)$", line)
                s = re.match(r"^STUDY scored (\d+)/(\d+)$", line)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
                    await self._send_study(
                        {
                            "t": "study_status",
                            "deck": deck_name,
                            "hash": h,
                            "phase": "games",
                            "done": done,
                            "total": total,
                        }
                    )
                elif line == "STUDY phase games done":
                    self._primers_cache = None
                    await self._send_study(
                        {
                            "t": "study_status",
                            "deck": deck_name,
                            "hash": h,
                            "phase": "games",
                            "done": total,
                            "total": total,
                        }
                    )
                elif s:
                    await self._send_study(
                        {
                            "t": "study_status",
                            "deck": deck_name,
                            "hash": h,
                            "phase": "scoring",
                            "done": int(s.group(1)),
                            "total": int(s.group(2)),
                        }
                    )
                elif line == "STUDY done":
                    self._primers_cache = None
                    await self._send_study(
                        {
                            "t": "study_status",
                            "deck": deck_name,
                            "hash": h,
                            "phase": "done",
                            "done": total,
                            "total": total,
                        }
                    )
        finally:
            try:
                Path(ydk_tmp).unlink(missing_ok=True)
            except OSError:
                pass
        rc = await proc.wait()
        if self._study_proc is proc:
            self._study_proc = None
        if rc != 0:
            print(f"ALERT web: study {deck_name} failed (exit {rc}): {last}", flush=True)
            await self._send_study(
                {
                    "t": "study_status",
                    "deck": deck_name,
                    "hash": h,
                    "phase": "failed",
                    "done": 0,
                    "total": total,
                    "error": last or f"exit {rc}",
                }
            )

    async def on_puzzle(self, msg: dict) -> None:
        from decklight.analysis.events import deck_path
        from decklight.decks import load_ydk
        from decklight.engine.duel import DuelConfig
        from decklight.engine.env import PyEnv
        from decklight.play.session import DuelSession

        pid = msg.get("id")
        found = next((p for p in self.get_puzzles() if str(p.get("id")) == str(pid)), None)
        if found is None:
            await self._error(f"unknown puzzle {pid!r}")
            return
        try:
            decks = [load_ydk(deck_path(d)) for d in found["decks"]]
        except (FileNotFoundError, ValueError, KeyError) as e:
            await self._error(f"puzzle decks unavailable: {e}")
            return
        try:
            seat = int(found["seat"])
            start_turn = int(found["turn"])
            prefix = [int(c) for c in found["prefix"]]
            game_seed = int(found.get("game_seed", found["seed"]))
        except (KeyError, TypeError, ValueError) as e:
            await self._error(f"bad puzzle {pid!r}: {e}")
            return
        cfg = DuelConfig(decks[0], decks[1], seed=int(found["seed"]))
        # Host-verified prefix decisions (what-if shape).
        env = PyEnv(self.engine, cfg, auto_single=False)
        env.reset()
        prefix_decisions: list[dict] = []
        prefix_lens: list[int] = []
        for i, c in enumerate(prefix):
            if env.done:
                break
            n = len(env.opts)
            prefix_lens.append(int(n))
            if n > 1:
                prefix_decisions.append({"player": int(env.player), "n_options": int(n)})
            env.step(int(c))
        if env.done:
            await self._error("the puzzle position is already game over")
            return
        try:
            product_prefix = [int(c) for c, n in zip(prefix, prefix_lens) if int(n) > 1]
        except ValueError as e:
            await self._error(f"replay failed: {e}")
            return
        try:
            opp = await asyncio.to_thread(self._make_opponent, "fast")
        except Exception as e:  # noqa: BLE001
            await self._error(f"could not start the fast opponent: {e}")
            return
        launcher = opp if hasattr(opp, "start_duel") else None
        bot = None if launcher is not None else opp
        meta = {
            "human_deck": str(found["decks"][seat]) if seat < len(found["decks"]) else "puzzle",
            "bot_deck": str(found["decks"][1 - seat]) if len(found["decks"]) > 1 else "puzzle",
            "mode": "fast",
            "practice": True,
            "puzzle": {"id": str(found["id"])},
            "hints_used": 0,
        }
        session = DuelSession(
            self.engine,
            cfg,
            seat,
            bot=bot,
            launcher=launcher,
            seed=game_seed,
            record_root=self.record_root,
            human_auto=True,
            record_meta=meta,
            driver_prefix=product_prefix,
            driver_prefix_decisions=prefix_decisions,
        )
        try:
            await asyncio.to_thread(session.reset)
        except Exception as e:  # noqa: BLE001
            session.close()
            await self._error(f"could not start the puzzle duel: {e}")
            return
        try:
            for c in prefix:
                if session.env.done:
                    break
                session.env.step(int(c))
            if session.env.done:
                raise ValueError("the replay position is already game over")
            if int(session.env.duel.turn) != start_turn:
                raise ValueError("puzzle prefix does not end on its turn")
            session.prod_len = len(prefix_decisions)
            if session.driver is not None:
                session.check_driver_step(
                    len(prefix_decisions), (int(session.env.player), len(session.env.opts))
                )
        except Exception as e:  # noqa: BLE001 -- ALERT, keep the current game
            print(f"ALERT web: puzzle replay failed: {e}", flush=True)
            session.close()
            await self._error(f"replay failed: {e}")
            return
        self.gen += 1
        if self.session is not None:
            self._drop_session(self.session)
        self.session = session
        self.mode = "fast"
        self.practice = True
        self.hints_used = 0
        self._remember_decks(
            decks[seat],
            decks[1 - seat],
            found["decks"][seat] if seat < len(found["decks"]) else "puzzle",
            found["decks"][1 - seat] if len(found["decks"]) > 1 else "puzzle",
        )
        self.undos = 0
        self.human_decisions = []
        self.what_if = None
        self.puzzle = {
            "id": str(found["id"]),
            "start_turn": int(start_turn),
            "goal": str(found.get("goal", "")),
            "n_plays": int(found.get("n_plays", found.get("n_moves", 0))),
            "status": "playing",
        }
        self.prompt_id = 0
        self.log_sent = 0
        self.cue_sent = 0
        self.finished = False
        self.end_reason = ""
        self._last_sig = ()
        self.winchance = None
        self._wc_n = 0
        self.auto_passed = set()
        self.record_path = None
        if self._review_task is not None and not self._review_task.done():
            self._review_task.cancel()
            self._review_task = None
        from decklight.web.stats import DuelStats

        self.stats = DuelStats(session.human_seat)
        self.start_advance()

    async def on_puzzle_solution(self, msg: dict) -> None:
        from decklight.analysis.events import deck_path
        from decklight.decks import load_ydk
        from decklight.engine.duel import DuelConfig

        pid = msg.get("id")
        found = next((p for p in self.get_puzzles() if str(p.get("id")) == str(pid)), None)
        if found is None:
            await self._error(f"unknown puzzle {pid!r}")
            return
        try:
            from decklight.web.replay import replay_frames

            decks = [load_ydk(deck_path(d)) for d in found["decks"]]
            cfg = DuelConfig(decks[0], decks[1], seed=int(found["seed"]))
            prefix = [int(c) for c in found["prefix"]]
            sol = [int(c) for c in found["solution"]]
            names = tuple(_pretty_deck_name(str(d)) for d in found["decks"][:2])
            while len(names) < 2:
                names += ("?",)
            frames = await asyncio.to_thread(
                replay_frames,
                self.engine,
                cfg,
                prefix + sol,
                names,
                expect_winner=int(found["seat"]),
                auto_single=False,
            )
            frames = frames[len(prefix) :]
            if self.ws is None:
                return
            try:
                await self.ws.send_json({"t": "puzzle_solution", "id": str(found["id"]), "frames": frames})
            except (WebSocketDisconnect, RuntimeError):
                pass
        except RuntimeError as e:
            await self._error(f"puzzle solution failed: {e}")
            return
        except Exception as e:  # noqa: BLE001 -- fail loud, keep playing
            import traceback

            print(
                "ALERT web: puzzle_solution crashed:\n" + "".join(traceback.format_exception(e)), flush=True
            )
            await self._error(f"puzzle solution failed: {e!r} (see server log)")

    async def on_settings(self, msg: dict) -> None:
        has_chain = "chain" in msg
        has_wc = "winchance" in msg
        if not has_chain and not has_wc:
            await self._error("settings.chain must be auto|all")
            return
        if has_chain:
            chain = msg.get("chain")
            if chain not in ("auto", "all"):
                await self._error("settings.chain must be auto|all")
                return
            self.chain_mode = chain
        if has_wc:
            wc = msg.get("winchance")
            if not isinstance(wc, bool):
                await self._error("settings.winchance must be true|false")
                return
            self.winchance_on = wc
        s = self.session
        if (
            has_chain
            and self.chain_mode == "auto"
            and s is not None
            and not self.finished
            and s.need_human
            and not self.advancing
        ):
            self.start_advance()  # a quiet window may be waiting right now
        if has_wc and s is not None and not self.advancing:
            if self.winchance_on and not self.finished and s.need_human:
                self._compute_winchance()  # the switch was just turned on at a prompt
            await self.send_state()  # push the toggle now (null when off)

    def _review_cache_path(self) -> Path | None:
        if self.record_path is None:
            return None
        return self.record_path.parent / "review.json"

    async def _send_review_progress(self, done: int, total: int) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send_json({"t": "review_progress", "done": done, "total": total})
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def on_review(self) -> None:
        s = self.session
        if s is None or not self.finished or s.result is None:
            await self._error("no finished game to review (finish a game first)")
            return
        if self.mode == "random":
            await self._error("Review needs the full bot")
            return
        launcher = getattr(s, "launcher", None)
        if launcher is None or getattr(launcher, "value", None) is None:
            await self._error("Review needs the full bot")
            return
        cache = self._review_cache_path()
        if cache is not None and cache.is_file():
            try:
                out = json.loads(cache.read_text())
                if self.ws is not None:
                    await self.ws.send_json({"t": "review", **out})
                return
            except (OSError, ValueError):
                pass
        if self._review_task is not None and not self._review_task.done():
            return  # a review is already running
        ws = self.ws
        gen = self.gen
        self._review_task = asyncio.create_task(self._run_review(ws, gen))

    async def _run_review(self, ws, gen: int) -> None:
        import traceback

        s = self.session
        try:
            from decklight.analysis.review import review_game
            from decklight.eval.winchance import load_calib

            if self.record_path is None or s is None:
                await self._error("no finished game to review (finish a game first)")
                return
            rec = json.loads(self.record_path.read_text())
            cfg = s.cfg
            launcher = getattr(s, "launcher", None)
            try:
                calib = load_calib()
            except Exception as e:  # noqa: BLE001 -- map missing: fail loudly, keep playing
                print(f"ALERT web: review calib unavailable: {e}", flush=True)
                await self._error("review is unavailable (calibration missing)")
                return
            try:
                n_dec = len([c for c in [rec.get("choices", [])] if c is not None])
            except Exception:  # noqa: BLE001
                n_dec = 0
            loop = asyncio.get_running_loop()

            def prog(done: int, total: int) -> None:
                if ws is None:
                    return
                try:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_json({"t": "review_progress", "done": done, "total": total}), loop
                    )
                except (RuntimeError, WebSocketDisconnect):
                    pass

            out = await asyncio.to_thread(
                review_game, self.engine, cfg, rec, launcher, calib, max_steps=32, progress=prog
            )
            _ = n_dec
            cache = self._review_cache_path()
            if cache is not None:
                try:
                    cache.write_text(json.dumps(out) + "\n")
                except OSError:
                    pass
            if self.ws is not ws or gen != self.gen:
                return
            try:
                await ws.send_json({"t": "review", **out})
            except (WebSocketDisconnect, RuntimeError):
                pass
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- fail loud, keep the client usable
            print("ALERT web: review crashed:\n" + "".join(traceback.format_exception(e)), flush=True)
            await self._error(f"review failed: {e!r} (see server log)")

    async def on_review_frame(self, msg: dict) -> None:
        from decklight.engine.env import PyEnv
        from decklight.web.view import build_prompt, build_view

        s = self.session
        if s is None or not self.finished:
            await self._error("no finished game to review (finish a game first)")
            return
        try:
            k = int(msg.get("k"))
        except (TypeError, ValueError):
            await self._error("review_frame.k must be a decision number")
            return
        cache = self._review_cache_path()
        if cache is None or not cache.is_file():
            await self._error("no review yet (send review first)")
            return
        try:
            out = json.loads(cache.read_text())
        except (OSError, ValueError):
            await self._error("no review yet (send review first)")
            return
        decs = out.get("decisions", [])
        if not 0 <= k < len(decs):
            await self._error(f"unknown decision {k}")
            return
        if self.record_path is None:
            await self._error("no finished game to review (finish a game first)")
            return
        rec = json.loads(self.record_path.read_text())
        choices = [int(c) for c in rec["choices"]]
        hs = int(rec["human_seat"])
        # Replay to decision k's position to rebuild the board + options.
        from decklight.analysis.review import collect_human_decisions

        dlist = collect_human_decisions(self.engine, s.cfg, rec)
        if not 0 <= k < len(dlist):
            await self._error(f"unknown decision {k}")
            return
        env = PyEnv(self.engine, s.cfg, auto_single=False)
        env.reset()
        for pc in dlist[k]["prefix"]:
            if env.done:
                break
            env.step(int(pc))
        _ = choices
        view = build_view(env, hs)
        prompt = build_prompt(env, hs, 999)
        entry = decs[k]
        opts = (prompt or {}).get("options", []) if prompt else []
        if self.ws is None:
            return
        try:
            await self.ws.send_json(
                {"t": "review_frame", "k": k, "view": view, "options": opts, "entry": entry}
            )
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def on_hint(self) -> None:
        """Practice-mode "Ask the bot": top-3 policy options + lookahead estimates.

        Computed only from the human's observation (the review's fairness rule).
        Rejected outside practice mode; counted in the record (`hints_used`).
        """
        if not self.practice:
            await self._error("Hints are only in practice mode")
            return
        s = self.session
        if s is None or s.done or not s.need_human or self.advancing:
            await self._error("no prompt is waiting for you right now")
            return
        launcher = getattr(s, "launcher", None)
        if launcher is None or not (hasattr(launcher, "base_logits") or hasattr(launcher, "forward")):
            await self._error("Hints need the full bot")
            return
        calib = None
        if self.winchance_on:
            if self._wc_map is None:
                try:
                    from decklight.eval.winchance import load_calib

                    self._wc_map = load_calib()
                except Exception as e:  # noqa: BLE001 -- map missing: answer without estimates
                    print(f"ALERT web: hint calib unavailable: {e}", flush=True)
                    self._wc_map = False
            if self._wc_map is not False:
                calib = self._wc_map
        try:
            from decklight.analysis.hints import compute_hint
            from decklight.web.view import build_prompt

            out = await asyncio.to_thread(
                compute_hint, self.engine, s.cfg, s.env, s.human_seat, launcher, calib
            )
            prompt = build_prompt(s.env, s.human_seat, self.prompt_id)
            labels = {o["index"]: o for o in (prompt["options"] if prompt else [])}
            options = [
                {
                    "index": int(o["index"]),
                    "label": str(labels.get(int(o["index"]), {}).get("label", f"option {o['index']}")),
                    "p": float(o["p"]),
                    "est": (float(o["est"]) if o["est"] is not None else None),
                }
                for o in out["options"]
            ]
            ms = float(out["ms"])
        except ValueError as e:
            await self._error(f"hint failed: {e}")
            return
        except Exception as e:  # noqa: BLE001 -- fail loud, keep playing
            import traceback

            print("ALERT web: hint crashed:\n" + "".join(traceback.format_exception(e)), flush=True)
            await self._error(f"hint failed: {e!r} (see server log)")
            return
        self.hints_used += 1
        s.record_meta["hints_used"] = self.hints_used
        if self.ws is None:
            return
        try:
            await self.ws.send_json({"t": "hint", "prompt": self.prompt_id, "options": options, "ms": ms})
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def _restart_from(
        self,
        prefix_host,
        *,
        mode: str,
        apply_option: int | None = None,
        what_if: dict | None = None,
        undos: int | None = None,
        hints_used: int = 0,
    ) -> bool:
        """Rebuild the live session from a host prefix (shared by what-if and undo).

        Replays `prefix_host` into a fresh `DuelSession` (the product driver
        replays the prefix verbatim, never re-searching it), swaps it in and
        starts `advance`. With `apply_option` set, that option is applied at the
        resulting prompt (what-if); otherwise the human's prompt at the prefix end
        is re-presented (undo). Every check is kept (driver lockstep divergence);
        on any failure it ALERTs and keeps the current session. Returns True on
        success.
        """
        from decklight.engine.env import PyEnv
        from decklight.play.session import DuelSession

        s = self.session
        if s is None:
            return False
        prefix_host = [int(c) for c in prefix_host]
        env = PyEnv(self.engine, s.cfg, auto_single=False)
        env.reset()
        prefix_decisions: list[dict] = []
        for c in prefix_host:
            if env.done:
                break
            if len(env.opts) > 1:
                prefix_decisions.append({"player": int(env.player), "n_options": len(env.opts)})
            env.step(int(c))
        if env.done:
            await self._error("the position is already game over")
            return False
        pre = (int(env.player), len(env.opts))
        turn = int(env.duel.turn)
        if what_if is not None and "turn" not in what_if:
            what_if = dict(what_if, turn=turn)
        from decklight.edopro.product_bot import host_to_product

        try:
            product_prefix = [int(c) for c in host_to_product(self.engine, s.cfg, prefix_host)]
        except ValueError as e:
            await self._error(f"replay failed: {e}")
            return False
        if len(product_prefix) != len(prefix_decisions):
            await self._error(
                f"replay failed: {len(prefix_decisions)} host decisions vs "
                f"{len(product_prefix)} product decisions"
            )
            return False
        try:
            opp = await asyncio.to_thread(self._make_opponent, mode)
        except Exception as e:  # noqa: BLE001
            await self._error(f"could not start the {mode} opponent: {e}")
            return False
        launcher = opp if hasattr(opp, "start_duel") else None
        bot = None if launcher is not None else opp
        meta = dict(s.record_meta or {})
        meta.update({"mode": mode, "practice": True, "what_if": what_if, "hints_used": hints_used})
        if undos is None:
            meta.pop("undos", None)
        else:
            meta["undos"] = int(undos)
        session = DuelSession(
            self.engine,
            s.cfg,
            s.human_seat,
            bot=bot,
            launcher=launcher,
            seed=s.seed,
            record_root=self.record_root,
            human_auto=True,
            record_meta=meta,
            driver_prefix=product_prefix,
            driver_prefix_decisions=prefix_decisions,
        )
        try:
            await asyncio.to_thread(session.reset)
        except Exception as e:  # noqa: BLE001
            session.close()
            await self._error(f"could not start the replay game: {e}")
            return False
        try:
            for c in prefix_host:
                if session.env.done:
                    break
                session.env.step(int(c))
            if session.env.done:
                raise ValueError("the replay position is already game over")
            if (int(session.env.player), len(session.env.opts)) != pre:
                raise ValueError(
                    f"replay diverged: host (player={session.env.player}, "
                    f"options={len(session.env.opts)}) vs prefix (player={pre[0]}, options={pre[1]})"
                )
            if apply_option is not None:
                if not 0 <= int(apply_option) < len(session.env.opts):
                    raise ValueError(f"illegal option {apply_option} of {len(session.env.opts)}")
                session.env.step(int(apply_option))
                session.prod_len = len(product_prefix) + 1
                if session.driver is not None:
                    session.driver.feed_human([int(apply_option)])
                    session.check_driver_step(len(product_prefix), pre)
            else:
                if not session.need_human:
                    raise ValueError("the rewind position is not a human prompt")
                session.prod_len = len(product_prefix)
                if session.driver is not None:
                    session.check_driver_step(len(product_prefix), pre)
        except Exception as e:  # noqa: BLE001 -- ALERT, keep the current game
            print(f"ALERT web: replay failed: {e}", flush=True)
            session.close()
            await self._error(f"replay failed: {e}")
            return False
        self.gen += 1
        if self.session is not None:
            self._drop_session(self.session)
        self.session = session
        self.practice = True
        self.hints_used = int(hints_used)
        self.what_if = what_if
        decks = (s.cfg.deck0, s.cfg.deck1)
        self._remember_decks(
            decks[s.human_seat],
            decks[1 - s.human_seat],
            meta.get("human_deck"),
            meta.get("bot_deck"),
        )
        self.prompt_id = 0
        self.log_sent = 0
        self.cue_sent = 0
        self.finished = False
        self.end_reason = ""
        self._last_sig = ()
        self.winchance = None
        self._wc_n = 0
        self.auto_passed = set()
        self.record_path = None
        if self._review_task is not None and not self._review_task.done():
            self._review_task.cancel()
            self._review_task = None
        from decklight.web.stats import DuelStats

        self.stats = DuelStats(session.human_seat)
        self.start_advance()
        return True

    async def on_whatif(self, msg: dict) -> None:
        """Start a new practice game from a reviewed decision with another option.

        Replays the record's choices up to human decision k, applies `option`,
        then continues live against the product bot. Valid after game over when
        a review exists.
        """
        s = self.session
        if s is None or not self.finished or s.result is None:
            await self._error("no finished game to review (finish a game first)")
            return
        cache = self._review_cache_path()
        if cache is None or not cache.is_file():
            await self._error("no review yet (send review first)")
            return
        try:
            k = int(msg.get("k"))
        except (TypeError, ValueError):
            await self._error("whatif.k must be a decision number")
            return
        try:
            option = int(msg.get("option"))
        except (TypeError, ValueError):
            await self._error("whatif.option must be an option number")
            return
        if self.record_path is None:
            await self._error("no finished game to review (finish a game first)")
            return
        rec = json.loads(self.record_path.read_text())
        from decklight.analysis.review import collect_human_decisions

        dlist = collect_human_decisions(self.engine, s.cfg, rec)
        if not 0 <= k < len(dlist):
            await self._error(f"unknown decision {k}")
            return
        mode = rec.get("mode", self.mode)
        if mode not in ("product", "fast"):
            await self._error("What-if needs the full bot")
            return
        prefix_host = [int(c) for c in dlist[k]["prefix"]]
        source = self.record_path.parent.name if self.record_path is not None else ""
        ok = await self._restart_from(
            prefix_host,
            mode=mode,
            apply_option=option,
            what_if={"source": source, "k": int(k), "option": int(option)},
            undos=None,
            hints_used=0,
        )
        if ok:
            self.human_decisions = []
            self.undos = 0
            self.puzzle = None
            try:
                if self.session is not None:
                    (self.session.record_meta or {}).pop("puzzle", None)
                    (self.session.record_meta or {}).pop("puzzle_result", None)
            except Exception:  # noqa: BLE001, S110 -- best effort
                pass

    async def on_undo(self) -> None:
        """Practice-mode undo: rewind to the human's previous real decision.

        Valid only in practice mode, while a human prompt waits (not while the bot
        thinks), with at least one real decision already made. Re-presents the
        previous real decision; the bot's choices before it are replayed, never
        re-searched (`_restart_from`). The record keeps its meta and gains
        `undos: <count>`, which carries across undos.
        """
        s = self.session
        if not self.practice:
            await self._error("Undo is only in practice mode")
            return
        if s is None or s.done:
            await self._error("no active game (send new_game first)")
            return
        if getattr(s, "launcher", None) is None:
            await self._error("Undo needs the full bot")
            return
        if self.advancing or s.thinking_since is not None:
            await self._error("Wait for the bot to finish")
            return
        if not s.need_human:
            await self._error("no prompt is waiting for you right now")
            return
        if not self.human_decisions:
            await self._error("Nothing to undo")
            return
        target = int(self.human_decisions[-1])
        prefix_host = [int(c) for c in s.env.choices[:target]]
        what_if = (s.record_meta or {}).get("what_if")
        n = int((s.record_meta or {}).get("undos", 0)) + 1
        # Auto-passes before the rewind point stay in the record (review's missed responses).
        kept_auto = {i for i in self.auto_passed if i <= target}
        ok = await self._restart_from(
            prefix_host,
            mode=(s.record_meta or {}).get("mode") or self.mode,
            apply_option=None,
            what_if=what_if,
            undos=n,
            hints_used=self.hints_used,
        )
        if ok:
            self.human_decisions.pop()
            self.undos = n
            self.auto_passed = kept_auto

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
    from decklight.engine.resources import load_cards
    from decklight.web.view import DB

    names = DB.names()
    text = DB.effect_text(code)
    cards = load_cards()
    cd = cards.get(code)
    st = DB.stats().get(code, (0, 0, 0, 0, 0, 0))
    atk, dfn, level, rank, link, type_ = st
    if cd is not None:
        atk, dfn = cd.attack, cd.defense
    return {
        "code": code,
        "name": names.get(code, f"#{code}"),
        "type": type_,
        "type_str": _type_str(type_),
        "attribute": cd.attribute if cd is not None else 0,
        "race": cd.race if cd is not None else 0,
        "level": level,
        "rank": rank,
        "link": link,
        "atk": atk,
        "def": dfn,
        "text": text,
    }


def _type_str(type_: int) -> str:
    from decklight.engine import constants as C

    if type_ & C.TYPE_MONSTER:
        kinds = []
        for bit, name in (
            (C.TYPE_FUSION, "Fusion"),
            (C.TYPE_SYNCHRO, "Synchro"),
            (C.TYPE_XYZ, "Xyz"),
            (C.TYPE_LINK, "Link"),
            (C.TYPE_RITUAL, "Ritual"),
            (C.TYPE_PENDULUM, "Pendulum"),
            (C.TYPE_TUNER, "Tuner"),
            (C.TYPE_EFFECT, "Effect"),
            (C.TYPE_NORMAL, "Normal"),
        ):
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


def create_app(
    *,
    token: str,
    engine=None,
    make_opponent=None,
    record_root: Path = ROOT / "runs" / "web",
    pics_dir: Path = PICS_DIR,
    opponent_default: str = "product",
    device: str = "cpu",
    replay_threads: int = 1,
) -> Starlette:
    _register_stack_dump()
    srv = Server(
        token=token,
        engine=engine,
        make_opponent=make_opponent,
        record_root=record_root,
        pics_dir=pics_dir,
        opponent_default=opponent_default,
        device=device,
        replay_threads=replay_threads,
    )

    def need_token(req) -> bool:
        return req.query_params.get("t") == srv.token

    async def index(req):
        if not need_token(req):
            return Response("unauthorized", status_code=401)
        page = ROOT / "web" / "index.html"
        if page.is_file():
            return Response(page.read_text(encoding="utf-8"), media_type="text/html")
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
        from decklight.engine.resources import extra_card_dirs

        for d in extra_card_dirs():
            cand = d / "pics" / f"{code}.jpg"
            if cand.is_file():
                return Response(cand.read_bytes(), media_type="image/jpeg")
        try:
            data = await asyncio.to_thread(_fetch_img, IMG_URLS[size].format(code=code))
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
        srv.winchance_on = False  # the switch is per connection, off by default
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
                if t == "review":
                    await srv.on_review()
                    continue
                if t == "review_frame":
                    await srv.on_review_frame(msg if isinstance(msg, dict) else {})
                    continue
                if t == "hint":  # read-only: answered from the waiting prompt
                    await srv.on_hint()
                    continue
                if t == "puzzles":  # read-only list
                    await srv.on_puzzles()
                    continue
                if t == "primer":  # read-only deck primer
                    await srv.on_primer(msg if isinstance(msg, dict) else {})
                    continue
                if t == "study":  # background deck study (spawns a subprocess)
                    await srv.on_study(msg if isinstance(msg, dict) else {})
                    continue
                if t == "puzzle_solution":  # read-only replay frames
                    await srv.on_puzzle_solution(msg if isinstance(msg, dict) else {})
                    continue
                async with srv.lock:
                    if t == "new_game":
                        await srv.on_new_game(msg)
                    elif t == "puzzle":
                        await srv.on_puzzle(msg if isinstance(msg, dict) else {})
                    elif t == "whatif":
                        await srv.on_whatif(msg if isinstance(msg, dict) else {})
                    elif t == "undo":
                        await srv.on_undo()
                    elif t == "choose":
                        await srv.on_choose(msg)
                    elif t == "resync":
                        if srv.session is None:
                            await ws.send_json({"t": "lobby", "decks": srv.list_decks()})
                        else:
                            await srv.send_state(full_log=True)
                    else:
                        await srv._error(f"unknown message {t!r}")
        finally:
            if srv.ws is ws:
                srv.ws = None

    routes = [
        Route("/", index),
        Route("/card/{code}", card),
        Route("/img/{code}.jpg", img),
        WebSocketRoute("/ws", ws_handler),
    ]
    web_dir = ROOT / "web"
    if web_dir.is_dir():  # client code is public; game data (ws, img, card) needs the token
        routes.append(Mount("/", StaticFiles(directory=web_dir)))

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        yield
        await srv.shutdown()  # kill a running deck study, if any

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.server = srv
    return app
