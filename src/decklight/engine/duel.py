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
from typing import NamedTuple, Protocol, Self

from decklight.engine import constants as C
from decklight.engine import messages as M
from decklight.engine import ocgapi as api
from decklight.engine import query as Q
from decklight.engine.knowledge import HandMemory
from decklight.engine.resources import CORE_LIB, CardData, index_scripts, load_cards

TCG_FLAGS = C.DUEL_MODE_MR5 | C.DUEL_TCG_SEGOC_NONPUBLIC
_MASK64 = (1 << 64) - 1


class DeckLike(Protocol):
    main: Sequence[int]
    extra: Sequence[int]


class Placement(NamedTuple):
    """A card put in place before the duel starts (puzzle start). `team` owns and controls it."""

    team: int
    code: int
    location: int  # one LOCATION_* (not DECK/EXTRA: those come from the decks)
    sequence: int = 0  # zone for MZONE/SZONE; list position elsewhere (appended in order)
    position: int = 0  # 0 = the default for the location (DEFAULT_POSITION)


DEFAULT_POSITION = {
    C.LOCATION_HAND: C.POS_FACEDOWN,
    C.LOCATION_MZONE: C.POS_FACEUP_ATTACK,
    C.LOCATION_SZONE: C.POS_FACEDOWN,
    C.LOCATION_GRAVE: C.POS_FACEUP,
    C.LOCATION_REMOVED: C.POS_FACEUP,
}


@dataclass(frozen=True)
class DuelConfig:
    deck0: DeckLike  # team 0 moves first
    deck1: DeckLike
    seed: int
    flags: int = TCG_FLAGS
    lp: int = 8000
    hand: int = 5
    draw: int = 1
    # Puzzle start: cards placed before the start (use hand=0 to skip the opening draw). The deck is
    # still shuffled by the seed, so a puzzle's hidden information is sampled by varying the seed.
    setup: tuple[Placement, ...] = ()
    # Shuffle each main deck (seeded) before it reaches the core, like EDOPro's host does (generic_duel.cpp
    # std::shuffle): the core itself never shuffles at duel start. False = the .ydk order, which every record
    # made before 2026-10-04 used (opening hand = the last 5 listed cards, usually traps/hand traps).
    shuffle: bool = True

    def main_of(self, team: int) -> list[int]:
        """Team `team`'s main deck in the order it is added to the duel (the top card is the last one)."""
        main = list((self.deck0 if team == 0 else self.deck1).main)
        if self.shuffle:
            shuffle_main(main, self.seed, team)
        return main


_M64 = (1 << 64) - 1


def shuffle_main(main: list[int], seed: int, team: int) -> None:
    """In-place seeded Fisher-Yates over a splitmix64 stream; bit-identical to `shuffled_main` in
    cpp/ygoenv/vecenv.cpp, so VecEnv games replay in PyEnv/CppEnv."""
    st = (seed ^ (0x9E3779B97F4A7C15 * (team + 1))) & _M64
    for k in range(len(main), 1, -1):
        st = (st + 0x9E3779B97F4A7C15) & _M64
        z = st
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
        z ^= z >> 31
        j = z % k
        main[k - 1], main[j] = main[j], main[k - 1]


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
        self.turn_player = 0
        self.first_player: int | None = None
        self.phase = 0
        self._ended = False
        # Card-knowledge memory (docs/obs_spec.md v2 annex): per hand seq, whether the other
        # player knows the card's identity. Seeded with puzzle-setup hands (no messages announce
        # them); everything else arrives via MSG_MOVE / DRAW / CONFIRM_CARDS / SHUFFLE_HAND.
        setup_hands: list[list[int]] = [[], []]
        for pl in config.setup:
            if pl.location == C.LOCATION_HAND:
                setup_hands[pl.team].append(pl.code)
        self.memory = HandMemory((setup_hands[0], setup_hands[1]))
        self.knowledge_resyncs = 0  # guard rebuilds; must stay 0 on clean replays (tests assert it)

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
            self._add_cards(team, C.LOCATION_DECK, config.main_of(team))
            self._add_cards(team, C.LOCATION_EXTRA, deck.extra)
        for pl in config.setup:
            self._place(pl)
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

    def _place(self, pl: Placement) -> None:
        if pl.location not in DEFAULT_POSITION:
            raise ValueError(f"cannot place a card in location {pl.location:#x}")
        info = api.OCG_NewCardInfo()
        info.team = info.con = pl.team
        info.duelist, info.code, info.loc, info.seq = 0, pl.code, pl.location, pl.sequence
        info.pos = pl.position or DEFAULT_POSITION[pl.location]
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
                if not self.memory.note(raw):
                    self.knowledge_resyncs += 1
                t = raw[0]
                if t == C.MSG_RETRY:
                    last = self.responses[-1].hex() if self.responses else None
                    raise InvalidResponse(f"rejected response {last} (step {len(self.responses)})")
                if t == C.MSG_NEW_TURN:
                    self.turn += 1
                    self.turn_player = raw[1] & 1
                    if self.first_player is None:
                        self.first_player = self.turn_player
                elif t == C.MSG_NEW_PHASE:
                    self.phase = M.parse_one(raw).phase
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
        if self.pending is not None:
            self._guard_knowledge()
        return self.pending

    def _guard_knowledge(self) -> None:
        """Resync guard: at every decision the tracked hand lengths must match the core's."""
        sizes = tuple(
            len([c for c in self.query_location(p, C.LOCATION_HAND) if c is not None]) for p in (0, 1)
        )
        if not self.memory.guard(sizes):
            self.knowledge_resyncs += 1

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

    # --- board queries ---

    def query_location(
        self, controller: int, location: int, flags: int = Q.CARD_FLAGS
    ) -> list[Q.CardState | None]:
        """Every slot of one location (None = empty zone)."""
        info = api.OCG_QueryInfo(flags, controller, location, 0, 0)
        length = ctypes.c_uint32()
        ptr = self.engine.lib.OCG_DuelQueryLocation(self._handle, ctypes.byref(length), ctypes.byref(info))
        return Q.parse_location(ctypes.string_at(ptr, length.value), controller, location)

    def query_field(self) -> bytes:
        length = ctypes.c_uint32()
        ptr = self.engine.lib.OCG_DuelQueryField(self._handle, ctypes.byref(length))
        return ctypes.string_at(ptr, length.value)

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
