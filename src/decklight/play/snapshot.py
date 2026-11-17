"""Offline positions rebuilt from a recorded EDOPro client stream.

A network client records every STOC_GAME_MSG it received (the same filtered core messages a
`StreamView` consumes). This module replays those messages through `StreamState` (no decisions:
SELECTs carry no board state) and snapshots the start of each opponent turn where the client
controls a face-down Spell/Trap. `build_config` turns one snapshot into a `DuelConfig` puzzle
start our engine can play (public cards exact, hidden opponent cards sampled).

The mapping is deliberately approximate (the sampled position is not the true hidden state):
- the opponent is engine team 0 (the first player), so the first turn is the opponent's and the
  measured turn is turn 3: the audit passes turns 1-2 with a no-op prefix so the opponent can
  attack (`turn 3` is their second turn, battle phase enabled). The real turn number is not restored;
- `lp` is the client's LP for both teams (the engine has one LP setting);
- the opponent's hidden hand/face-downs/deck/extra are drawn from the codes the stream revealed,
  weighted by how often they were revealed ("padded with copies");
- hidden monsters go to hidden monster zones and hidden spells/traps to hidden S/T zones;
- counters, lingering effects and once-per-duel state cannot be recovered from the stream.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from decklight.engine import constants as C
from decklight.engine.duel import DuelConfig, Placement
from decklight.play.stream import StreamState

PLACEMENT_LOCATIONS = (
    C.LOCATION_HAND,
    C.LOCATION_MZONE,
    C.LOCATION_SZONE,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
)
LIST_LOCATIONS = (
    C.LOCATION_HAND,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
    C.LOCATION_EXTRA,
)


class SkipPosition(RuntimeError):
    """The snapshot cannot be turned into an engine start; `reason` is a short counter key."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class SnapshotCard:
    owner: int  # absolute stream seat
    code: int  # 0 = hidden from the client
    location: int
    sequence: int
    position: int


@dataclass(frozen=True)
class Snapshot:
    stream: str
    turn: int
    sky_seat: int
    lp: tuple[int, int]  # absolute stream seats
    deck_count: tuple[int, int]  # absolute stream seats
    cards: tuple[SnapshotCard, ...]


@dataclass
class BuiltPosition:
    config: DuelConfig
    sample: int
    notes: dict = field(default_factory=dict)


def read_stream(path: str | Path) -> tuple[dict, list[dict]]:
    """A recorded `.stream.jsonl.gz`: (header, frames)."""
    with gzip.open(path, "rt") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    if not lines or "main" not in lines[0]:
        raise SkipPosition("stream_no_header")
    return lines[0], lines[1:]


def _start_seat(frames: list[dict]) -> int:
    for fr in frames:
        if fr.get("dir") != "in" or fr.get("proto") != 1 or not fr.get("hex"):
            continue
        raw = bytes.fromhex(fr["hex"])
        if raw and raw[0] == C.MSG_START:
            return int(raw[1]) & 0xF if len(raw) >= 2 else 0
    raise SkipPosition("stream_no_start")


def _has_facedown_st(state: StreamState, seat: int) -> bool:
    for card in state.slots[(seat, C.LOCATION_SZONE)]:
        if card is not None and card.position & C.POS_FACEDOWN:
            return True
    return False


def _snapshot_from_state(state: StreamState, seat: int, stream: str) -> Snapshot:
    cards: list[SnapshotCard] = []
    for owner in (0, 1):
        for loc, slots in ((C.LOCATION_MZONE, state.slots), (C.LOCATION_SZONE, state.slots)):
            for seq, card in enumerate(slots[(owner, loc)]):
                if card is None:
                    continue
                if card.overlay:
                    raise SkipPosition("overlay_material")
                cards.append(SnapshotCard(owner, card.code, loc, seq, card.position))
        for loc in LIST_LOCATIONS:
            for seq, card in enumerate(state.lists[(owner, loc)]):
                if card.overlay:
                    raise SkipPosition("overlay_material")
                cards.append(SnapshotCard(owner, card.code, loc, seq, card.position))
    return Snapshot(
        stream=stream,
        turn=state.turn,
        sky_seat=seat,
        lp=(int(state.lp[0]), int(state.lp[1])),
        deck_count=(int(state.deck_count[0]), int(state.deck_count[1])),
        cards=tuple(cards),
    )


def extract_snapshots(
    header: dict, frames: list[dict], cards_db: dict, stream: str = "", skipped: dict | None = None
) -> list[Snapshot]:
    """Every opponent-turn start where the client has a face-down Spell/Trap.

    `skipped`, when given, is a counter dict: a position the engine cannot express is counted
    there and skipped (otherwise the exception propagates).
    """
    seat = _start_seat(frames)
    state = StreamState(list(header["main"]), list(header.get("extra", ())), cards_db, seat)
    started = False
    out: list[Snapshot] = []
    for fr in frames:
        if fr.get("dir") != "in" or fr.get("proto") != 1 or not fr.get("hex"):
            continue
        raw = bytes.fromhex(fr["hex"])
        if not raw:
            continue
        if raw[0] == C.MSG_START:
            started = True
        if not started:
            continue
        state.feed(raw)
        if raw[0] == C.MSG_NEW_TURN and state.turn_player != seat and _has_facedown_st(state, seat):
            try:
                out.append(_snapshot_from_state(state, seat, stream))
            except SkipPosition as e:
                if skipped is None:
                    raise
                skipped[e.reason] = skipped.get(e.reason, 0) + 1
    return out


def stream_seat(path: str | Path) -> int:
    _, frames = read_stream(path)
    return _start_seat(frames)


def _stable_seed(*parts) -> int:
    return int(hashlib.md5("/".join(str(p) for p in parts).encode()).hexdigest()[:15], 16)


def _sample(pool: Counter[int], rng: random.Random) -> int:
    codes = list(pool)
    weights = [pool[c] for c in codes]
    return int(rng.choices(codes, weights=weights, k=1)[0])


def _zone_position(card: SnapshotCard) -> int:
    if card.location == C.LOCATION_MZONE:
        return card.position if card.position in (1, 2, 4, 8) else C.POS_FACEUP_ATTACK
    if card.location == C.LOCATION_SZONE:
        return C.POS_FACEUP if card.position & C.POS_FACEUP else C.POS_FACEDOWN
    if card.location == C.LOCATION_HAND:
        return 0
    return C.POS_FACEUP


def _pools(
    snapshot: Snapshot, opp: int, cards_db: dict
) -> tuple[Counter[int], Counter[int], Counter[int], Counter[int]]:
    """Revealed opponent codes split by broad type: all, monsters, spells/traps, extra-deck monsters."""
    seen = Counter(c.code for c in snapshot.cards if c.owner == opp and c.code and c.code in cards_db)
    monsters: Counter[int] = Counter()
    spells: Counter[int] = Counter()
    extra: Counter[int] = Counter()
    for code, n in seen.items():
        d = cards_db[code]
        if d.type & C.TYPE_MONSTER:
            monsters[code] = n
            if d.type & (C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK):
                extra[code] = n
        else:
            spells[code] = n
    # Extra-deck monsters already visible in the opponent's extra deck are the best prior.
    extra.update(
        c.code
        for c in snapshot.cards
        if c.owner == opp and c.location == C.LOCATION_EXTRA and c.code and c.code in cards_db
    )
    return seen, monsters, spells, extra


def build_config(
    snapshot: Snapshot,
    header: dict,
    cards_db: dict,
    *,
    sample: int = 0,
    seed: int = 0,
    opp_hand: tuple[int, ...] = (),
    opp_deck_add: tuple[int, ...] = (),
    opp_extra_add: tuple[int, ...] = (),
) -> BuiltPosition:
    """One `DuelConfig` puzzle start for `snapshot` (hidden opponent cards sampled).

    Recorded-turn starts: `opp_hand` fills the opponent's hidden hand first (extra codes
    become extra hand cards: they used more cards than they held at the turn start, the rest came from the draw or
    searches), `opp_deck_add` / `opp_extra_add` are put into the deck / fill hidden extra-deck slots first (cards the
    recorded turn used from there), and all of them join the sampling pool."""
    sky, opp = snapshot.sky_seat, 1 - snapshot.sky_seat
    rng = random.Random(_stable_seed(seed, snapshot.stream, snapshot.turn, sample))
    sky_lp = int(snapshot.lp[sky])
    if sky_lp <= 0:
        raise SkipPosition("sky_dead")

    main = Counter(int(c) for c in header["main"])
    extra = [int(c) for c in header.get("extra", ())]
    for code in extra:
        if code not in cards_db:
            raise SkipPosition("sky_card_not_in_db")

    sky_places: list[Placement] = []
    sky_extra: list[int] = []
    for c in snapshot.cards:
        if c.owner != sky:
            continue
        if c.location == C.LOCATION_EXTRA:
            if not c.code or c.code not in cards_db:
                raise SkipPosition("sky_extra_hidden")
            sky_extra.append(c.code)
            continue
        if not c.code or c.code not in cards_db:
            raise SkipPosition("sky_card_not_in_db")
        if c.location in PLACEMENT_LOCATIONS:
            sky_places.append(Placement(1, c.code, c.location, c.sequence, _zone_position(c)))
            if c.location != C.LOCATION_EXTRA and main.get(c.code, 0) > 0:
                main[c.code] -= 1

    pool_all, pool_mon, pool_spell, pool_extra = _pools(snapshot, opp, cards_db)
    for code in (*opp_hand, *opp_deck_add, *opp_extra_add):
        if code in cards_db:
            pool_all[code] += 1
            (pool_mon if cards_db[code].type & C.TYPE_MONSTER else pool_spell)[code] += 1
    hand_left, extra_left = [c for c in opp_hand if c in cards_db], [c for c in opp_extra_add if c in cards_db]
    if not pool_all:
        raise SkipPosition("no_opp_revealed")

    opp_places: list[Placement] = []
    opp_extra: list[int] = []
    for c in snapshot.cards:
        if c.owner != opp:
            continue
        if c.location == C.LOCATION_EXTRA:
            if c.code:
                opp_extra.append(c.code)
            elif extra_left:
                opp_extra.append(extra_left.pop(0))
            else:
                pool = pool_extra or pool_mon or pool_all
                opp_extra.append(_sample(pool, rng))
            continue
        if c.code:
            code = c.code
        elif c.location == C.LOCATION_HAND and hand_left:
            code = hand_left.pop(0)
        elif c.location == C.LOCATION_MZONE:
            code = _sample(pool_mon or pool_all, rng)
        elif c.location == C.LOCATION_SZONE:
            code = _sample(pool_spell or pool_all, rng)
        else:
            code = _sample(pool_all, rng)
        if code not in cards_db:
            raise SkipPosition("opp_sample_not_in_db")
        opp_places.append(Placement(0, code, c.location, c.sequence, _zone_position(c)))

    n_hand = sum(1 for c in snapshot.cards if c.owner == opp and c.location == C.LOCATION_HAND)
    for k, code in enumerate(hand_left):  # more used than held: extra hand cards
        opp_places.append(Placement(0, code, C.LOCATION_HAND, n_hand + k, 0))
    opp_extra += extra_left
    add = [c for c in opp_deck_add if c in cards_db]
    opp_deck = add + [_sample(pool_all, rng) for _ in range(max(0, snapshot.deck_count[opp] - len(add)))]
    sky_main = [code for code, n in main.items() for _ in range(max(0, n))]
    sky_extra_final = sky_extra
    if not sky_main:
        raise SkipPosition("sky_deck_empty")

    config = DuelConfig(
        deck0=_Deck(tuple(opp_deck), tuple(opp_extra)),
        deck1=_Deck(tuple(sky_main), tuple(sky_extra_final)),
        seed=seed + sample,
        lp=sky_lp,
        hand=0,
        draw=0,
        setup=tuple(opp_places + sky_places),
        shuffle=True,
    )
    notes = {
        "opp_deck": len(opp_deck),
        "sky_deck": len(sky_main),
        "sky_deck_stream": int(snapshot.deck_count[sky]),
        "opp_extra": len(opp_extra),
        "sky_extra": len(sky_extra_final),
        "pool_codes": len(pool_all),
        "lp": [int(snapshot.lp[0]), int(snapshot.lp[1])],
        "opp_hand_forced": len(opp_hand), "opp_hand_extra": len(hand_left),
    }
    return BuiltPosition(config=config, sample=sample, notes=notes)


class _Deck:
    """Minimal `DeckLike` for sampled lists."""

    __slots__ = ("extra", "main")

    def __init__(self, main: tuple[int, ...], extra: tuple[int, ...]) -> None:
        self.main, self.extra = main, extra
