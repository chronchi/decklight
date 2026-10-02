"""Reference duel driver over the ctypes bindings (the correctness oracle for the C++ env).

A duel is fully determined by `DuelConfig` + the list of responses, so a game can be stored as a
replay and rebuilt exactly (`Duel.replay`). That is also how states get cloned for search.

    engine = Engine()
    duel = engine.new_duel(DuelConfig(deck0, deck1, seed=1))
    while (msg := duel.advance()) is not None:
        duel.respond(agent.respond(msg))
    duel.winner
"""

import ctypes
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

from ygo_zero.engine import constants as C
from ygo_zero.engine import messages as M
from ygo_zero.engine import ocgapi as api
from ygo_zero.engine.resources import CORE_LIB, CardData, index_scripts, load_cards

TCG_FLAGS = C.DUEL_MODE_MR5 | C.DUEL_TCG_SEGOC_NONPUBLIC
_MASK64 = (1 << 64) - 1


class DeckLike(Protocol):
    main: Sequence[int]
    extra: Sequence[int]


@dataclass(frozen=True)
class DuelConfig:
    deck0: DeckLike  # team 0 moves first
    deck1: DeckLike
    seed: int
    flags: int = TCG_FLAGS
    lp: int = 8000
    hand: int = 5
    draw: int = 1


class InvalidResponse(RuntimeError):
    """The core answered MSG_RETRY: the last response was illegal."""


class ScriptError(RuntimeError):
    pass


def seed_words(seed: int) -> list[int]:
    """Expand a 64-bit seed into the core's 4x64-bit xoshiro state with splitmix64 (never all zero)."""
    out, x = [], seed & _MASK64
    for _ in range(4):
        x = (x + 0x9E3779B97F4A7C15) & _MASK64
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        out.append(z ^ (z >> 31))
    return out


_HINTS = frozenset({C.MSG_CARD_HINT, C.MSG_PLAYER_HINT})


def canonical_trace(trace: Sequence[bytes]) -> list[bytes]:
    """The trace with each run of consecutive CARD_HINT/PLAYER_HINT messages sorted.

    The core emits those UI hints while iterating pointer-keyed containers (e.g. effect resets), so
    their relative order follows heap addresses and differs between otherwise identical duels in one
    process (NOTEBOOK 2026-10-01). Everything else is deterministic; compare traces through this.
    """
    out: list[bytes] = []
    run: list[bytes] = []
    for raw in trace:
        if raw[0] in _HINTS:
            run.append(raw)
            continue
        out += sorted(run) + [raw]
        run = []
    return out + sorted(run)


class Engine:
    """Loaded core + card data + script index; creates duels. One per process is enough."""

    SYSTEM_SCRIPTS = ("constant.lua", "utility.lua")  # the core never requests these itself

    def __init__(
        self,
        lib_path: str | Path = CORE_LIB,
        cards: dict[int, CardData] | None = None,
        scripts: dict[str, Path] | None = None,
    ) -> None:
        self.lib = api.load_library(lib_path)
        self.cards = cards if cards is not None else load_cards()
        self._script_paths = scripts if scripts is not None else index_scripts()
        self._script_cache: dict[str, bytes] = {}
        self._setcodes = (ctypes.c_uint16 * 5)()
        self._card_reader = api.OCG_DataReader(self._read_card)
        self._card_reader_done = api.OCG_DataReaderDone(lambda payload, data: None)

    def _read_card(self, payload, code, data_ptr) -> None:
        cd = self.cards.get(code)
        if cd is None:
            return  # zeroed data: the core treats it as an unknown card
        d = data_ptr.contents
        d.code, d.alias, d.type, d.level = cd.code, cd.alias, cd.type, cd.level
        d.attribute, d.race, d.attack, d.defense = cd.attribute, cd.race, cd.attack, cd.defense
        d.lscale, d.rscale, d.link_marker = cd.lscale, cd.rscale, cd.link_marker
        if cd.setcodes:
            for i, s in enumerate(cd.setcodes):
                self._setcodes[i] = s
            self._setcodes[len(cd.setcodes)] = 0
            d.setcodes = self._setcodes  # copied by the core before cardReaderDone

    def script(self, name: str) -> bytes | None:
        if name not in self._script_cache:
            path = self._script_paths.get(name)
            if path is None:
                return None
            self._script_cache[name] = path.read_bytes()
        return self._script_cache[name]

    def new_duel(self, config: DuelConfig) -> "Duel":
        return Duel(self, config)


class Duel:
    def __init__(self, engine: Engine, config: DuelConfig) -> None:
        self.engine, self.config, lib = engine, config, engine.lib
        self.responses: list[bytes] = []
        self.trace: list[bytes] = []  # every raw message, in order (determinism golden)
        self.log: list[tuple[int, str]] = []
        self.pending: M.Select | None = None
        self.winner: int | None = None  # 0/1, 2 = draw; None while running
        self.win_reason = 0
        self.turn = 0
        self._ended = False

        # Per-duel callbacks so script errors land in this duel's log.
        self._script_reader = api.OCG_ScriptReader(self._read_script)
        self._log_handler = api.OCG_LogHandler(
            lambda payload, s, t: self.log.append((t, s.decode(errors="replace")))
        )
        opts = api.OCG_DuelOptions()
        for i, w in enumerate(seed_words(config.seed)):
            opts.seed[i] = w
        opts.flags = config.flags
        for team in (opts.team1, opts.team2):
            team.startingLP, team.startingDrawCount, team.drawCountPerTurn = (
                config.lp,
                config.hand,
                config.draw,
            )
        opts.cardReader, opts.cardReaderDone = engine._card_reader, engine._card_reader_done
        opts.scriptReader, opts.logHandler = self._script_reader, self._log_handler
        opts.enableUnsafeLibraries = 0

        self._handle = api.OCG_Duel()
        status = lib.OCG_CreateDuel(ctypes.byref(self._handle), ctypes.byref(opts))
        if status != 0:
            raise RuntimeError(f"OCG_CreateDuel failed: {api.CREATION_STATUS[status]}")
        for name in Engine.SYSTEM_SCRIPTS:
            if not self._load_script(name):
                raise ScriptError(f"cannot load {name}: {self.log[-3:]}")
        for team, deck in ((0, config.deck0), (1, config.deck1)):
            self._add_cards(team, C.LOCATION_DECK, deck.main)
            self._add_cards(team, C.LOCATION_EXTRA, deck.extra)
        lib.OCG_StartDuel(self._handle)

    # --- callbacks / setup ---

    def _load_script(self, name: str) -> bool:
        data = self.engine.script(name)
        if data is None:
            return False
        return self.engine.lib.OCG_LoadScript(self._handle, data, len(data), name.encode()) == 1

    def _read_script(self, payload, duel, name: bytes) -> int:
        data = self.engine.script(name.decode())
        if data is None:
            return 0
        return int(self.engine.lib.OCG_LoadScript(duel, data, len(data), name) == 1)

    def _add_cards(self, team: int, location: int, codes: Sequence[int]) -> None:
        info = api.OCG_NewCardInfo()
        info.team = info.con = team
        info.duelist, info.loc, info.pos = 0, location, C.POS_FACEDOWN_DEFENSE
        for seq, code in enumerate(codes):
            info.code, info.seq = code, seq
            self.engine.lib.OCG_DuelNewCard(self._handle, ctypes.byref(info))

    # --- stepping ---

    @property
    def ended(self) -> bool:
        return self._ended

    def advance(self) -> M.Select | None:
        """Run the core until it needs a response (returned) or the duel ends (None)."""
        lib, length = self.engine.lib, ctypes.c_uint32()
        while self.pending is None and not self._ended:
            status = lib.OCG_DuelProcess(self._handle)
            ptr = lib.OCG_DuelGetMessage(self._handle, ctypes.byref(length))
            buf = ctypes.string_at(ptr, length.value) if ptr and length.value else b""
            for raw in M.split_buffer(buf):
                self.trace.append(raw)
                t = raw[0]
                if t == C.MSG_RETRY:
                    last = self.responses[-1].hex() if self.responses else None
                    raise InvalidResponse(f"rejected response {last} (step {len(self.responses)})")
                if t == C.MSG_NEW_TURN:
                    self.turn += 1
                elif t == C.MSG_WIN:
                    win = M.parse_one(raw)
                    self.winner, self.win_reason = win.player, win.reason
                    self._ended = True  # the core may never report END after a win; stop here
                    break
                elif t in M.SELECT_TYPES:
                    self.pending = M.parse_one(raw)
            if status == api.DUEL_STATUS_END:
                self._ended = True
            elif status == api.DUEL_STATUS_AWAITING and self.pending is None and not self._ended:
                raise RuntimeError("core is awaiting a response but sent no select message")
        return self.pending

    def respond(self, response: bytes) -> None:
        if self.pending is None:
            raise RuntimeError("no pending selection")
        self.engine.lib.OCG_DuelSetResponse(self._handle, response, len(response))
        self.responses.append(response)
        self.pending = None

    def run(self, agents: Sequence[Callable[[M.Select], bytes]], max_steps: int = 20_000) -> int | None:
        """Play to the end with one policy per player. Returns the winner (None if max_steps hit)."""
        for _ in range(max_steps):
            msg = self.advance()
            if msg is None:
                return self.winner
            self.respond(agents[msg.player](msg))
        return None

    def close(self) -> None:
        if self._handle:
            self.engine.lib.OCG_DestroyDuel(self._handle)
            self._handle = api.OCG_Duel()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @classmethod
    def replay(cls, engine: Engine, config: DuelConfig, responses: Sequence[bytes]) -> "Duel":
        """Rebuild a duel from its config + responses; the result is ready for the next response."""
        duel = cls(engine, config)
        for r in responses:
            if duel.advance() is None:
                raise ValueError(f"duel ended after {len(duel.responses)} of {len(responses)} responses")
            duel.respond(r)
        duel.advance()
        return duel
