"""EDOPro replay export: a game (DuelConfig + raw core responses) as an old-format `.yrp` that EDOPro opens from
its Replay menu and re-simulates with its own core and card scripts (edopro/gframe/old_replay_mode.cpp).

Layout (edopro/gframe/replay.{h,cpp}, generic_duel.cpp): ExtendedReplayHeader (72 bytes: id 'yrp1', client
version, flags, timestamp, datasize, hash, props[8], header_version 1, xoshiro seed[4]), then, uncompressed:
1 + name (40 bytes UTF-16) per side, lp/hand/draw (u32), duel flags (u64), per player main and extra deck
(u32 count + codes in the order EDOPro adds them), rule cards (u32 0), then every response (u8 length + bytes).

EDOPro adds every card with sequence 0 (push to the top); our engine passes the list index as the sequence, so
the core puts the second card at the bottom (field::add_card: sequence 1 = deck bottom). `edopro_main` turns our
list into EDOPro's so the deck ends up in the same order. `verify` rebuilds the duel exactly as EDOPro does and
checks that the core produces the same message stream (up to card order inside a message, see `verify`), so a file that passes plays out the same game in an EDOPro
with the same core and scripts (third_party/ygopro-core, third_party/CardScripts); a different EDOPro version
can drift from the point where a script or the core differs.
"""

from __future__ import annotations

import ctypes
import struct
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from ygo_zero.engine import constants as C
from ygo_zero.engine.duel import Duel, DuelConfig, Engine, canonical_trace, seed_words

REPLAY_YRP1 = 0x31707279
REPLAY_LUA64, REPLAY_NEWREPLAY, REPLAY_64BIT_DUELFLAG, REPLAY_EXTENDED_HEADER = 0x10, 0x20, 0x100, 0x200
CLIENT_VERSION = 41 | (0 << 8) | (11 << 16) | (0 << 24)  # EDOPro 41.0, core 11.0 (edopro/gframe/config.h)
_HEADER = struct.Struct("<6I8sQ4Q")


def edopro_main(main: Sequence[int]) -> list[int]:
    """Our add order -> EDOPro's (all at sequence 0) giving the same deck: index 1 went to the bottom."""
    m = list(main)
    return [m[1], m[0], *m[2:]] if len(m) > 1 else m


def _name(s: str) -> bytes:
    b = s[:19].encode("utf-16-le")
    return b + b"\0" * (40 - len(b))


def to_yrp(
    config: DuelConfig, responses: Sequence[bytes], names: tuple[str, str], timestamp: int | None = None
) -> bytes:
    if config.setup:
        raise ValueError("puzzle starts (setup) cannot be exported")
    body = bytearray()
    for n in names:
        body += struct.pack("<I", 1) + _name(n)
    body += struct.pack("<3IQ", config.lp, config.hand, config.draw, config.flags)
    for team, deck in ((0, config.deck0), (1, config.deck1)):
        main, extra = edopro_main(config.main_of(team)), list(deck.extra)
        body += struct.pack(f"<I{len(main)}I", len(main), *main) + struct.pack(
            f"<I{len(extra)}I", len(extra), *extra
        )
    body += struct.pack("<I", 0)
    for r in responses:
        if not 0 < len(r) < 256:
            raise ValueError(f"response of {len(r)} bytes does not fit the u8 length")
        body += bytes([len(r)]) + r
    flags = REPLAY_LUA64 | REPLAY_NEWREPLAY | REPLAY_64BIT_DUELFLAG | REPLAY_EXTENDED_HEADER
    ts = int(time.time()) if timestamp is None else timestamp
    head = _HEADER.pack(
        REPLAY_YRP1, CLIENT_VERSION, flags, ts, len(body), 0, b"\0" * 8, 1, *seed_words(config.seed)
    )
    return head + bytes(body)


@dataclass
class Yrp:
    names: list[str]
    seed: list[int]
    lp: int
    hand: int
    draw: int
    flags: int
    decks: list[tuple[list[int], list[int]]]
    responses: list[bytes]


def parse_yrp(data: bytes) -> Yrp:
    """Read a file written by `to_yrp` the way edopro/gframe/replay.cpp does (uncompressed, 1 vs 1 only)."""
    rid, _ver, flags, _ts, _size, _hash, _props, hver, *seed = _HEADER.unpack_from(data)
    if rid != REPLAY_YRP1 or hver != 1 or not flags & REPLAY_EXTENDED_HEADER or flags & 0x1:
        raise ValueError(f"not an uncompressed extended yrp1 (id {rid:#x}, flags {flags:#x}, header v{hver})")
    pos = _HEADER.size

    def u(fmt: str):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]

    names = []
    for _ in range(2):
        if u("I") != 1:
            raise ValueError("only 1 vs 1 replays")
        names.append(data[pos : pos + 40].decode("utf-16-le").split("\0")[0])
        pos += 40
    lp, hand, draw, dflags = u("3IQ")

    def codes() -> list[int]:
        nonlocal pos
        n = u("I")
        out = list(struct.unpack_from(f"<{n}I", data, pos))
        pos += 4 * n
        return out

    decks = []
    for _ in range(2):
        main = codes()
        decks.append((main, codes()))
    if u("I") != 0:
        raise ValueError("rule cards are not supported")
    responses = []
    while pos < len(data):
        n = data[pos]
        responses.append(bytes(data[pos + 1 : pos + 1 + n]))
        pos += 1 + n
    return Yrp(names, list(seed), lp, hand, draw, dflags, decks, responses)


class _EdoproDuel(Duel):
    """A Duel set up the way EDOPro's old replay mode does: every card added at sequence 0, in file order."""

    def _add_cards(self, team: int, location: int, codes: Sequence[int]) -> None:
        from ygo_zero.engine import ocgapi as api

        info = api.OCG_NewCardInfo()
        info.team = info.con = team
        info.duelist, info.loc, info.pos, info.seq = 0, location, C.POS_FACEDOWN_DEFENSE, 0
        for code in codes:
            info.code = code
            self.engine.lib.OCG_DuelNewCard(self._handle, ctypes.byref(info))


def play_responses(duel: Duel, responses: Sequence[bytes]) -> int:
    """Feed `responses` in order; returns how many were consumed before the duel ended."""
    n = 0
    while duel.advance() is not None:
        if n == len(responses):
            break
        duel.respond(responses[n])
        n += 1
    return n


def verify(engine: Engine, data: bytes, original: Duel) -> str | None:
    """None if the file, replayed as EDOPro sets it up, reproduces `original` (a finished duel) message for
    message; else what differs."""
    y = parse_yrp(data)
    if y.seed != seed_words(original.config.seed):
        return "seed words differ"
    cfg = DuelConfig(SimpleNamespace(main=y.decks[0][0], extra=y.decks[0][1]),
                     SimpleNamespace(main=y.decks[1][0], extra=y.decks[1][1]), original.config.seed,
                     flags=y.flags, lp=y.lp, hand=y.hand, draw=y.draw, shuffle=False)  # fmt: skip
    duel = _EdoproDuel(engine, cfg)
    try:
        try:
            n = play_responses(duel, y.responses)
        except Exception as e:  # noqa: BLE001 - the core rejected a response: the games diverged
            return f"{type(e).__name__}: {e}"
        if n != len(y.responses):
            return f"duel ended after {n} of {len(y.responses)} responses"
        if not duel.ended:
            return f"duel still running after all {n} responses"
        if duel.winner != original.winner:
            return f"winner {duel.winner} != original {original.winner}"
        a, b = canonical_trace(duel.trace), canonical_trace(original.trace)
        if len(a) != len(b):
            return f"{len(a)} messages != original {len(b)}"
        for k, (x, z) in enumerate(zip(a, b)):
            # Allowed: the same message with its bytes in another order. EDOPro creates the main deck in list
            # order while we create its first two cards the other way round, so those two swap creation ids and
            # can swap places in a card list the core sorts by id (NOTEBOOK 2026-10-16). Any response that
            # picked one of them would make the game diverge, which the rest of the stream would show.
            if x != z and (len(x) != len(z) or x[0] != z[0] or sorted(x) != sorted(z)):
                return f"message {k} of {len(b)} differs (type {z[0]})"
    finally:
        duel.close()
    return None


def edopro_engine(install: Path) -> Engine:
    """An Engine with the core, scripts and card data a local EDOPro install uses (ProjectIgnis/: the updates repo's
    bin/libocgcore.so and script/ over the base script/), to check a file the way that EDOPro will play it."""
    from ygo_zero.engine.resources import index_scripts, load_cards

    repo = install / "repositories/delta-bagooska"
    scripts = index_scripts(install / "script")
    scripts.update(index_scripts(repo / "script"))
    cdbs = sorted((install / "expansions").glob("*.cdb")) + sorted(repo.glob("*.cdb"))
    lib = repo / "bin/libocgcore.so"
    return Engine(lib if lib.is_file() else install / "libocgcore.so", load_cards(cdbs), scripts)
