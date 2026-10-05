"""Observation arrays for the player to act (docs/obs_spec.md). Reference implementation for the C++ env."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ygo_zero.engine import constants as C
from ygo_zero.engine import query as Q
from ygo_zero.engine.actions import Decision, Kind, Option
from ygo_zero.engine.messages import CardRef

OBS_VERSION = 1
MAX_CARDS, CARD_COLS = 160, 21
MAX_CHAIN, CHAIN_COLS = 16, 5
MAX_OPTIONS, ACTION_COLS = 64, 10
HISTORY, HISTORY_COLS = 32, 13
HISTORY_MAX = 512  # long salient window (obs v2, docs/obs_spec.md); v1 HISTORY stays the reference
HLOG_COLS = 14  # history_log: stored option row (10) + actor, turn, phase, public
GLOBAL_COLS = 24

# Option kinds that carry no information for future decisions: PASS answers and zone picks
# (MSG_SELECT_PLACE / DISFIELD). Auto-taken single options are already never recorded.
SALIENT_SKIP = frozenset({Kind.PASS, Kind.PLACE})  # == {11, 19}; row col 0 is the int kind

STATUS_OBS_MASK = (
    C.STATUS_DISABLED | C.STATUS_SET_TURN | C.STATUS_SUMMON_TURN | C.STATUS_FLIP_SUMMON_TURN
    | C.STATUS_SPSUMMON_TURN | C.STATUS_FORM_CHANGED | C.STATUS_ATTACK_CANCELED | C.STATUS_CHAINING
    | C.STATUS_FORBIDDEN
)  # fmt: skip

LOC_ID = {
    C.LOCATION_DECK: 1, C.LOCATION_HAND: 2, C.LOCATION_MZONE: 3, C.LOCATION_SZONE: 4,
    C.LOCATION_GRAVE: 5, C.LOCATION_REMOVED: 6, C.LOCATION_EXTRA: 7, C.LOCATION_OVERLAY: 8,
}  # fmt: skip
_MZONE_POS = {
    C.POS_FACEUP_ATTACK: 1,
    C.POS_FACEDOWN_ATTACK: 2,
    C.POS_FACEUP_DEFENSE: 3,
    C.POS_FACEDOWN_DEFENSE: 4,
}
_ROW_LOCS = (
    C.LOCATION_HAND,
    C.LOCATION_MZONE,
    C.LOCATION_SZONE,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
    C.LOCATION_EXTRA,
)


def i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def loc_id(location: int) -> int:
    return LOC_ID[C.LOCATION_OVERLAY] if location & C.LOCATION_OVERLAY else LOC_ID.get(location, 0)


def ref_key(ref: CardRef) -> tuple[int, int, int, int]:
    """Row key of a card as messages name it. Overlay units: (holder con, OVERLAY, holder seq, index)."""
    if ref.location & C.LOCATION_OVERLAY:
        return (ref.controller, C.LOCATION_OVERLAY, ref.sequence, ref.position)
    return (ref.controller, ref.location, ref.sequence, 0)


@dataclass(slots=True)
class Row:
    card: Q.CardState
    visible: bool  # identity known to the viewer


class Board:
    """All cards as seen by `me`, in obs row order, with a lookup from message card refs to rows."""

    def __init__(self, duel, me: int) -> None:
        self.me = me
        q = {(p, loc): duel.query_location(p, loc) for p in (0, 1) for loc in Q.QUERY_LOCATIONS}
        self.counts = {k: sum(c is not None for c in v) for k, v in q.items()}
        field = duel.query_field()
        self.lp = Q.lps(field)
        self.chain = Q.parse_chain(field)
        rows: list[Row] = []
        for p in (me, 1 - me):
            mine = p == me
            for loc in _ROW_LOCS:
                for c in q[(p, loc)]:
                    if c is None:
                        continue
                    if loc == C.LOCATION_EXTRA and not mine and not c.position & C.POS_FACEUP:
                        continue
                    rows.append(Row(c, mine or self._public(c)))
            for c in q[(p, C.LOCATION_MZONE)]:
                for i, code in enumerate(c.overlay if c else ()):
                    rows.append(
                        Row(
                            Q.CardState(
                                p, C.LOCATION_OVERLAY, c.sequence, code, public=True, overlay_index=i
                            ),
                            True,
                        )
                    )
        deck = sorted((c for c in q[(me, C.LOCATION_DECK)] if c), key=lambda c: (c.code, c.sequence))
        rows += [Row(c, True) for c in deck]
        self.truncated = max(0, len(rows) - MAX_CARDS)
        self.rows = rows[:MAX_CARDS]
        self.index: dict[tuple[int, int, int, int], int] = {}
        for i, r in enumerate(self.rows):
            c = r.card
            self.index[(c.controller, c.location, c.sequence, c.overlay_index)] = i

    @staticmethod
    def _public(c: Q.CardState) -> bool:
        if c.location in (C.LOCATION_GRAVE, C.LOCATION_OVERLAY):
            return True
        return c.public or bool(c.position & C.POS_FACEUP and c.location != C.LOCATION_HAND)

    def row_of(self, ref: CardRef | None) -> int:
        return -1 if ref is None else self.index.get(ref_key(ref), -1)

    def option_code(self, opt: Option) -> int:
        """The code the viewer may see for an option's card."""
        if opt.card is None:
            return opt.code
        r = self.row_of(opt.card)
        if r >= 0:
            return opt.code if self.rows[r].visible else 0
        # a card offered from a hidden pile (Deck, face-down Extra Deck) is revealed to the chooser by the effect
        # (e.g. looking at the opponent's Extra Deck); face-down field cards are rows above and stay hidden
        if opt.card.controller == self.me or opt.card.location in (C.LOCATION_DECK, C.LOCATION_EXTRA):
            return opt.code
        return 0

    def visible_codes(self) -> set[int]:
        return {r.card.code for r in self.rows if r.visible and r.card.code}

    # --- arrays ---

    def cards(self, selected: Sequence[CardRef] = ()) -> tuple[np.ndarray, np.ndarray]:
        out = np.zeros((MAX_CARDS, CARD_COLS), np.int32)
        mask = np.zeros(MAX_CARDS, bool)
        for i, r in enumerate(self.rows):
            out[i] = self._card_row(r)
            mask[i] = True
        for ref in selected:
            if (i := self.row_of(ref)) >= 0:
                out[i, 20] = 1
        return out, mask

    def knowledge(self, memory) -> tuple[np.ndarray, np.ndarray]:
        """Card-knowledge memory (obs v2, docs/obs_spec.md): per card row, `card_known` uint8 [160]
        (1 = opponent's hidden card whose identity I remember, 2 = my own hidden card the
        opponent knows, else 0) and `known_code` int32 [160] (the remembered passcode for
        `card_known == 1` rows, else 0)."""
        known = np.zeros(MAX_CARDS, np.uint8)
        code = np.zeros(MAX_CARDS, np.int32)
        for i, r in enumerate(self.rows):
            c = r.card
            if c.location != C.LOCATION_HAND:
                continue
            hk = memory.known[c.controller]
            if not (0 <= c.sequence < len(hk) and hk[c.sequence]):
                continue
            if c.controller == self.me:
                known[i] = 2  # my own hand is visible to me; the flag is about the opponent
            elif not r.visible:
                known[i] = 1
                code[i] = c.code
        return known, code

    def _card_row(self, r: Row) -> list[int]:
        c, me = r.card, self.me
        loc = c.location
        if loc == C.LOCATION_MZONE:
            pos = _MZONE_POS.get(c.position, 0)
        elif loc in (C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_OVERLAY):
            pos = 0
        else:
            pos = 5 if c.position & C.POS_FACEUP else 6
        seq = 0 if loc == C.LOCATION_DECK else c.sequence + 1
        public = int(self._public(c))
        base = [0, LOC_ID[loc], seq, c.controller ^ me, 0, pos, public]
        if not r.visible:
            return base + [0] * 14
        base[0], base[4] = c.code, c.owner ^ me if loc != C.LOCATION_OVERLAY else c.controller ^ me
        race = (c.race & 0xFFFFFFFF) | ((c.race >> 62 & 1) << 31)
        return base + [
            c.type, c.attribute, i32(race), c.level, c.rank, c.link, c.link_marker,
            c.attack, c.defense, c.base_attack, c.base_defense,
            c.lscale | c.rscale << 8, c.status & STATUS_OBS_MASK, 0,
        ]  # fmt: skip

    def chain_array(self) -> tuple[np.ndarray, np.ndarray]:
        out = np.zeros((MAX_CHAIN, CHAIN_COLS), np.int32)
        mask = np.zeros(MAX_CHAIN, bool)
        for i, ch in enumerate(self.chain[:MAX_CHAIN]):
            ref = CardRef(ch.code, ch.controller, ch.location, ch.sequence, ch.position)
            out[i] = [self.row_of(ref), ch.code, i32(ch.description >> 20), ch.description & 0xFFFFF,
                      ch.triggering_controller ^ self.me]  # fmt: skip
            mask[i] = True
        return out, mask

    def global_array(self, duel, msg_type: int) -> np.ndarray:
        me, opp = self.me, 1 - self.me
        g = [self.lp[me], self.lp[opp], duel.turn, duel.phase, int(duel.first_player == me),
             int(duel.turn_player == me)]  # fmt: skip
        for p in (me, opp):
            g += [self.counts[(p, loc)] for loc in Q.QUERY_LOCATIONS]
        g += [len(self.chain), msg_type, self.truncated, OBS_VERSION]
        return np.array(g, np.int32)

    def option_row(self, opt: Option, msg_type: int, step: int) -> list[int]:
        ref = opt.card
        ctrl = -1 if ref is None else ref.controller ^ self.me
        loc = 0 if ref is None else loc_id(ref.location)
        return [int(opt.kind), self.row_of(ref), self.option_code(opt), i32(opt.desc >> 20), opt.desc & 0xFFFFF,
                i32(opt.arg), msg_type, loc, ctrl, step]  # fmt: skip


def actions_array(board: Board, dec: Decision, opts: Sequence[Option]) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((MAX_OPTIONS, ACTION_COLS), np.int32)
    mask = np.zeros(MAX_OPTIONS, bool)
    for i, o in enumerate(opts[:MAX_OPTIONS]):
        out[i] = board.option_row(o, dec.msg.type, dec.step)
        mask[i] = True
    return out, mask


# Taking these reveals the card to the opponent.
_REVEALING = frozenset({Kind.ACTIVATE, Kind.SUMMON, Kind.SPSUMMON, Kind.REPOS, Kind.ATTACK})


@dataclass(slots=True)
class HistoryEntry:
    row: list[int]  # option row from the actor's view (card col set to -1)
    actor: int  # absolute player
    turn: int
    phase: int
    public: bool  # does the other player see the card identity?


def history_entry(board: Board, opt: Option, msg_type: int, step: int, turn: int, phase: int) -> HistoryEntry:
    row = board.option_row(opt, msg_type, step)
    row[1] = -1
    if opt.kind in _REVEALING:
        public = True
    elif opt.card is None:
        public = not (opt.kind == Kind.POSITION and opt.arg & C.POS_FACEDOWN)
    else:
        r = board.row_of(opt.card)
        public = opt.card.controller != board.me or (r >= 0 and Board._public(board.rows[r].card))
    return HistoryEntry(row, board.me, turn, phase, public)


def _history_row(e: HistoryEntry, me: int, turn: int) -> list[int]:
    """One recorded entry as seen by `me`: the single transform both history windows share."""
    row = list(e.row)
    if row[8] >= 0:
        row[8] ^= e.actor ^ me
    if e.actor != me and not e.public:
        row[2] = row[3] = row[4] = 0
    return row + [e.actor ^ me, turn - e.turn, e.phase]


def history_array(
    entries: Sequence[HistoryEntry], me: int, turn: int, n: int = HISTORY, salient: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """The most recent ≤ n entries, most-recent-first. `salient` skips PASS / PLACE entries (obs v2
    `hist_long`); the default reproduces the v1 `history` window exactly."""
    if salient:
        entries = [e for e in entries if e.row[0] not in SALIENT_SKIP]
    out = np.zeros((n, HISTORY_COLS), np.int32)
    mask = np.zeros(n, bool)
    for i, e in enumerate(reversed(entries[-n:])):
        out[i] = _history_row(e, me, turn)
        mask[i] = True
    return out, mask


def history_log(entries: Sequence[HistoryEntry]) -> np.ndarray:
    """All recorded entries in absolute form [n, 14]: cols 0-9 the stored row (col 1 = -1, col 8 in the
    actor's frame), 10 actor (absolute seat), 11 turn, 12 phase, 13 public (0/1). Shards store this log
    and rebuild windows from it (`bc_data.history_window`)."""
    out = np.zeros((len(entries), HLOG_COLS), np.int32)
    for i, e in enumerate(entries):
        out[i, :ACTION_COLS] = e.row
        out[i, 10] = e.actor
        out[i, 11] = e.turn
        out[i, 12] = e.phase
        out[i, 13] = int(e.public)
    return out
