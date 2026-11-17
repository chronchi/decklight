"""The client's seat: rebuilt from the filtered EDOPro message stream (plan §3.1).

A network client only receives the STOC_GAME_MSG payloads our `DuelHost` sends the
human seat: `build_start`, filtered core messages, and MSG_UPDATE_DATA / MSG_UPDATE_CARD
refresh packets. `StreamState` maintains a `BoardSource` from those alone (plus our deck
list and `cards.cdb`); `StreamView` builds decisions with `make_decision` and observations
with the shared `obs.build_obs`, so the trained policy plays as a client.

Seats are absolute engine seats everywhere in the stream: `build_start`'s seat byte is the
viewer's absolute seat and every controller/location reference is untranslated (verified
against `edopro/filter.py`, which compares the raw player byte to the viewer seat).
"""

from __future__ import annotations

import struct
from collections import Counter

from decklight.engine import constants as C
from decklight.engine import messages as M
from decklight.engine import query as Q
from decklight.engine.actions import Decision, Kind, Option, make_decision
from decklight.engine.events import public_events
from decklight.engine.knowledge import HandMemory
from decklight.engine.messages import CardRef
from decklight.engine.obs import (
    MAX_OPTIONS,
    STATUS_OBS_MASK,
    Board,
    HistoryEntry,
    build_obs,
    history_entry,
)
from decklight.engine.query import CardState

# Fixed-size zones (core containers); HAND/GRAVE/REMOVED/EXTRA are ordered lists.
_SLOTS = {C.LOCATION_MZONE: 7, C.LOCATION_SZONE: 8}
_LISTS = (C.LOCATION_HAND, C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA)

# A SEARCH/show guarantee no client-side tracker can keep: the opponent's hand order after
# a shuffle (their SHUFFLE_HAND codes are zeroed) and identities the stream never carried.


def _loc_info(raw: bytes, off: int) -> tuple[int, int, int, int]:
    con, loc = raw[off], raw[off + 1]
    (seq,) = struct.unpack_from("<I", raw, off + 2)
    (pos,) = struct.unpack_from("<I", raw, off + 6)
    return con, loc, seq, pos


def deck_card_state(code: int, cards: dict, controller: int, sequence: int) -> CardState:
    """A full `CardState` for one of our deck-list cards, from the card DB.

    Mirrors what the core stores from the same DB rows (`resources.load_cards` + the
    OCG_DataReader) and reports through `get_info`: rank is the level byte of XYZ
    monsters, the link rating of LINK monsters, base stats are the DB atk/def (LINK def
    is the marker, base 0) — and level/attribute/race/attack/defense read 0 for
    non-monsters (`card::get_*` gate on TYPE_MONSTER).
    """
    d = cards[code]
    monster = bool(d.type & C.TYPE_MONSTER)
    lv = d.level & 0xFFFFFFFF
    return CardState(
        controller,
        C.LOCATION_DECK,
        sequence,
        code,
        position=0,
        type=d.type,
        level=0 if (d.type & (C.TYPE_XYZ | C.TYPE_LINK)) or not monster else lv,
        rank=lv if d.type & C.TYPE_XYZ else 0,
        attribute=d.attribute if monster else 0,
        race=d.race if monster else 0,
        attack=d.attack if monster else 0,
        defense=d.defense if monster else 0,
        base_attack=d.attack if monster else 0,
        base_defense=d.defense if monster else 0,
        owner=controller,
        status=0,
        public=False,
        lscale=d.lscale,
        rscale=d.rscale,
        link=lv if d.type & C.TYPE_LINK else 0,
        link_marker=d.link_marker,
    )


class StreamState:
    """A `BoardSource` maintained from the filtered message stream (absolute seats)."""

    def __init__(self, deck_main: list[int], deck_extra: list[int], cards: dict, seat: int) -> None:
        self.seat = seat
        self.cards_db = cards
        self.deck_list = Counter(deck_main)
        self.extra_list = list(deck_extra)
        self.slots: dict[tuple[int, int], list[CardState | None]] = {
            (p, loc): [None] * n for p in (0, 1) for loc, n in _SLOTS.items()
        }
        self.lists: dict[tuple[int, int], list[CardState]] = {(p, loc): [] for p in (0, 1) for loc in _LISTS}
        self.deck_count = [0, 0]
        # True sequences of our deck cards, learned from messages that name them
        # (search SELECTs list (code, seq); draws pop the top without naming it).
        self.deck_seqs: dict[int, int] = {}
        # Masked status bits of returned-to-deck cards by sequence (a summoned monster
        # bounced to the deck keeps SUMMON_TURN; turn-scoped, cleared on NEW_TURN).
        self.deck_stat: dict[int, int] = {}
        # Unpositioned (code, status) pairs known to sit in our deck: shuffles and
        # grave-swaps preserve cards but not positions, so positional knowledge is
        # demoted here and re-attached when a message names the sequence again.
        self.deck_loose: Counter[tuple[int, int]] = Counter()
        self.turn = 0
        self.phase = 0
        self.first_player = None
        self.turn_player = 0
        self.lp = [8000, 8000]
        self.chain: list[Q.ChainLink] = []
        self.memory = HandMemory()

    # --- BoardSource surface ---

    def query_location(self, player: int, location: int, flags: int = Q.CARD_FLAGS):
        if (player, location) in self.slots:
            return list(self.slots[(player, location)])
        if (player, location) in self.lists:
            return list(self.lists[(player, location)])
        if location == C.LOCATION_DECK:
            if player == self.seat:
                return self._own_deck()
            return [CardState(player, location, i, owner=player) for i in range(self.deck_count[player])]
        raise KeyError(f"untracked location {location:#x}")

    def field_state(self) -> tuple[tuple[int, int], list[Q.ChainLink]]:
        return (self.lp[0], self.lp[1]), list(self.chain)

    def _own_deck(self) -> list[CardState]:
        """Our deck rows: the deck list minus our known cards elsewhere (multiset).

        Cards whose true deck sequence a message revealed keep it, so refs to them
        map to the right row; the rest fill the remaining sequences in code order
        (same per-code index sets as the engine's (code, seq) sort: equal codes
        render identically).
        """
        elsewhere: Counter[int] = Counter()
        for (p, _), cards in {**self.slots, **self.lists}.items():
            if p != self.seat:
                continue
            for c in cards:
                if c is not None and c.code:
                    elsewhere[c.code] += 1
                    elsewhere.update(c.overlay)
        unknown_own = 0  # our cards held by the opponent whose code this seat never saw
        for (p, _), cards in self.lists.items():
            if p == self.seat:
                continue
            for c in cards:  # our cards held by the opponent with known codes
                if c.owner == self.seat:
                    if c.code:
                        elsewhere[c.code] += 1
                    else:
                        unknown_own += 1
        for (p, _), cards in self.slots.items():
            if p == self.seat:
                continue
            for c in cards:
                if c is not None and c.owner == self.seat:
                    if c.code:
                        elsewhere[c.code] += 1
                    else:
                        unknown_own += 1
        rest = self.deck_list - elsewhere
        size = max(0, sum(rest.values()) - unknown_own)
        known: dict[int, int] = {}
        stat: dict[int, int] = {}
        for seq, code in self.deck_seqs.items():
            if seq < size and rest.get(code, 0) > 0:
                known[seq] = code
                rest[code] -= 1
                if seq in self.deck_stat:
                    stat[seq] = self.deck_stat[seq]
        free = [s for s in range(size) if s not in known]
        fill = sorted(rest.elements())
        known.update(zip(free, fill))
        loose = Counter(self.deck_loose)  # local copy: queries never mutate state
        out = []
        for seq, code in sorted(known.items(), key=lambda kv: (kv[1], kv[0])):
            c = deck_card_state(code, self.cards_db, self.seat, seq)
            if seq in stat:
                c.status = stat[seq]
            else:
                for lcode, lstat in list(loose):
                    if lcode == code:
                        c.status = lstat
                        loose[(lcode, lstat)] -= 1
                        if loose[(lcode, lstat)] <= 0:
                            del loose[(lcode, lstat)]
                        break
            out.append(c)
        return out

    def learn_deck_seqs(self, refs) -> None:
        """Record true deck sequences named by a message (search SELECTs)."""
        for r in refs:
            if r.controller == self.seat and r.location == C.LOCATION_DECK and r.code:
                self.deck_seqs[r.sequence] = r.code

    # --- packet feed ---

    def feed(self, raw: bytes) -> None:
        """Fold one STOC_GAME_MSG payload in (SELECTs for our seat carry no state)."""
        if not raw:
            return
        if not self.memory.note(raw):
            pass  # a shuffle-length mismatch rebuilt a list (client-side resync, like the guard)
        t = raw[0]
        if t == C.MSG_START:
            self._start(raw)
        elif t == C.MSG_UPDATE_DATA:
            self._update_data(raw)
        elif t == C.MSG_UPDATE_CARD:
            self._update_card(raw)
        elif t == C.MSG_MOVE:
            self._move(raw)
        elif t == C.MSG_DRAW:
            self._draw(raw)
        elif t == C.MSG_NEW_TURN:
            self.turn += 1
            self.turn_player = raw[1] & 1
            if self.first_player is None:
                self.first_player = self.turn_player
            self.deck_stat.clear()  # summon/set-turn bits are turn-scoped
            self.deck_loose.clear()
        elif t == C.MSG_NEW_PHASE:
            (phase,) = struct.unpack_from("<H", raw, 1)
            self.phase = phase
        elif t in (C.MSG_DAMAGE, C.MSG_PAY_LPCOST):
            player = raw[1]
            (amount,) = struct.unpack_from("<I", raw, 2)
            self.lp[player] -= amount
        elif t == C.MSG_RECOVER:
            player = raw[1]
            (amount,) = struct.unpack_from("<I", raw, 2)
            self.lp[player] += amount
        elif t == C.MSG_LPUPDATE:
            player = raw[1]
            (lp,) = struct.unpack_from("<I", raw, 2)
            self.lp[player] = lp
        elif t == C.MSG_CHAINING:
            link = self._chain_link(raw)
            self.chain.append(link)
            # The handler joins the chain: the core flags it CHAINING and its
            # QUERY_IS_PUBLIC turns true. Status is display-zeroed for hidden cards
            # so it is safe to set by position; public shows even on hidden rows, so
            # on hands (whose order a shuffle can hide from us) it needs a code match.
            self.set_status(link.controller, link.location, link.sequence, C.STATUS_CHAINING, True)
            card = self._entry_at(link.controller, link.location, link.sequence)
            if card is not None and (link.location != C.LOCATION_HAND or card.code == link.code):
                self.set_public(link.controller, link.location, link.sequence, True)
        elif t == C.MSG_CHAIN_SOLVING:
            # Solve-start clears every handler's CHAINING silently (no message, no
            # refresh); no decision can sit between it and this message, so clearing
            # here is observationally exact. Nothing legitimately keeps the bit past
            # this point (later chains set it again with new CHAININGs).
            for cards in list(self.slots.values()) + list(self.lists.values()):
                for c in cards:
                    if c is not None:
                        c.status &= ~C.STATUS_CHAINING
            for seq in list(self.deck_stat):
                self.deck_stat[seq] &= ~C.STATUS_CHAINING
            for code, status in list(self.deck_loose):
                if status & C.STATUS_CHAINING:
                    n = self.deck_loose.pop((code, status))
                    rest = status & ~C.STATUS_CHAINING
                    if rest:
                        self.deck_loose[(code, rest)] += n
            # The resolving handler may be a monster placed as a Continuous Trap
            # (EFFECT_CHANGE_TYPE): the core reverts it when the activation resolves,
            # with no message. The engine view already shows its card-DB stats.
            if self.chain:
                self._revert_trap_monster(self.chain[-1])
        elif t == C.MSG_CHAIN_SOLVED:
            if self.chain:
                solved = self.chain.pop()
                # Chain relations release per link at this point (QUERY_IS_PUBLIC flips
                # back, except under continuous reveal effects, which the next refresh
                # re-latches). Grave/banished handlers have no covering refresh.
                self.set_public(solved.controller, solved.location, solved.sequence, False)
                # A monster placed as a Continuous Trap (EFFECT_CHANGE_TYPE) reverts to
                # its card-DB type when its activation resolves, with no refresh naming
                # it (trap-monster summons leave the zone and are refresh-covered).
                self._revert_trap_monster(solved)
        elif t == C.MSG_CHAIN_END:
            for ch in self.chain:
                self.set_public(ch.controller, ch.location, ch.sequence, False)
            self.chain.clear()
        elif t == C.MSG_POS_CHANGE:
            self._pos_change(raw)
        elif t == C.MSG_SWAP:
            self._swap(raw)
        elif t == C.MSG_REMOVE_CARDS:
            self._remove_cards(raw)
        elif t == C.MSG_SWAP_GRAVE_DECK:
            player = raw[1]
            self.deck_count[player] = len(self.lists[(player, C.LOCATION_GRAVE)])
            if player == self.seat:
                # The grave (public: true statuses) becomes the deck, order unknown.
                for c in self.lists[(player, C.LOCATION_GRAVE)]:
                    if c.code and c.status & STATUS_OBS_MASK:
                        self.deck_loose[(c.code, c.status & STATUS_OBS_MASK)] += 1
                self.deck_seqs.clear()
                self.deck_stat.clear()
            self.lists[(player, C.LOCATION_GRAVE)] = []
        elif t == C.MSG_SHUFFLE_DECK:
            if raw[1] == self.seat:
                self._demote_deck()
        elif t == C.MSG_REVERSE_DECK:
            self._demote_deck()  # both decks reverse; no player byte
        # MSG_WIN is tracked by the view; everything else carries no board state.

    def _start(self, raw: bytes) -> None:
        _, seat, lp0, lp1, n0, e0, n1, e1 = struct.unpack("<BBIIHHHH", raw[:18])
        assert seat == self.seat, f"START for seat {seat}, viewer {self.seat}"
        self.lp = [lp0, lp1]
        self.deck_count = [n0, n1]
        # The opponent's extra deck is never listed to us: face-down rows, so it counts like the engine's.
        for p, n in ((0, e0), (1, e1)):
            if p != self.seat and not self.lists[(p, C.LOCATION_EXTRA)]:
                self.lists[(p, C.LOCATION_EXTRA)] = [
                    CardState(p, C.LOCATION_EXTRA, i, owner=p, position=C.POS_FACEDOWN) for i in range(n)
                ]

    def _update_data(self, raw: bytes) -> None:
        player, loc = raw[1], raw[2]
        cards = Q.parse_location(raw[3:], player, loc)
        for c in cards:  # public servers query the extra deck without QUERY_LINK: the DB has it
            if c is not None and c.type & C.TYPE_LINK and not c.link and c.code in self.cards_db:
                d = deck_card_state(c.code, self.cards_db, player, c.sequence)
                c.link, c.link_marker = d.link, d.link_marker
        if (player, loc) in self.slots:
            self.slots[(player, loc)] = cards
        elif (player, loc) in self.lists:
            self.lists[(player, loc)] = [c for c in cards if c is not None]
            self._resync_chain_list(player, loc)
        # UPDATE_DATA never covers DECK (counts come from START/DRAW/MOVE).

    def _resync_chain_list(self, player: int, loc: int) -> None:
        """A refreshed list is authoritative (a shuffle reorders it): chain handlers are
        live card pointers, so move links with that code to their new index."""
        cards = self.lists.get((player, loc), [])
        used: set[int] = set()
        for i, ch in enumerate(self.chain):
            if (ch.controller, ch.location) != (player, loc) or not ch.code:
                continue
            cur = ch.sequence
            if 0 <= cur < len(cards) and cur not in used and cards[cur].code == ch.code:
                used.add(cur)  # duplicate codes: keep the link's live index when it still fits
                self.chain[i] = Q.ChainLink(
                    ch.code,
                    ch.controller,
                    ch.location,
                    cur,
                    cards[cur].position,
                    ch.triggering_controller,
                    ch.triggering_location,
                    ch.triggering_sequence,
                    ch.description,
                )
                continue
            for j, c in enumerate(cards):
                if j not in used and c.code == ch.code:
                    used.add(j)
                    self.chain[i] = Q.ChainLink(
                        ch.code,
                        ch.controller,
                        ch.location,
                        j,
                        c.position,
                        ch.triggering_controller,
                        ch.triggering_location,
                        ch.triggering_sequence,
                        ch.description,
                    )
                    break

    def _update_card(self, raw: bytes) -> None:
        player, loc, seq = raw[1], raw[2], raw[3]
        card = CardState(player, loc, seq, owner=player)
        Q.parse_card(raw[4:], 0, card)
        key = (player, loc)
        if key in self.slots:
            slots = self.slots[key]
            while len(slots) <= seq:
                slots.append(None)
            slots[seq] = card
        elif key in self.lists:
            lst = self.lists[key]
            while len(lst) <= seq:
                lst.append(CardState(player, loc, len(lst), owner=player))
            lst[seq] = card

    def _take_from(self, con: int, loc: int, seq: int) -> CardState | None:
        key = (con, loc & ~C.LOCATION_OVERLAY)
        if key in self.slots:
            slots = self.slots[key]
            if 0 <= seq < len(slots):
                card = slots[seq]
                slots[seq] = None
                return card
            return None
        if (con, loc) in self.lists:
            lst = self.lists[(con, loc)]
            if 0 <= seq < len(lst):
                card = lst.pop(seq)
                for i, c in enumerate(lst):
                    c.sequence = i
                # Chain handlers are live card pointers: the core reports their current
                # index, so later cards in the same list shift down with it.
                self._shift_chain(con, loc, seq + 1, -1)
                return card
            return None
        if loc == C.LOCATION_DECK and con in (0, 1):
            self.deck_count[con] = max(0, self.deck_count[con] - 1)
        return None

    def _fill_stats(self, card: CardState, force: bool = False) -> CardState:
        """Fill a card we know by code but whose live row we never saw (a place/move from
        hand or deck names only code and position). Identity fields are preserved; a later
        refresh replaces the row with the engine's live stats."""
        if not card.code or (card.type and not force) or card.code not in self.cards_db:
            return card
        d = deck_card_state(card.code, self.cards_db, card.controller, card.sequence)
        for field in (
            "type",
            "level",
            "rank",
            "attribute",
            "race",
            "attack",
            "defense",
            "base_attack",
            "base_defense",
            "lscale",
            "rscale",
            "link",
            "link_marker",
        ):
            setattr(card, field, getattr(d, field))
        return card

    def _put_to(self, card: CardState | None, code: int, con: int, loc: int, seq: int, pos: int) -> None:
        if loc == C.LOCATION_DECK:
            self.deck_count[con] += 1
            return
        old_loc = None if card is None else card.location
        old_seq = None if card is None else card.sequence
        if card is None:
            card = CardState(con, loc & ~C.LOCATION_OVERLAY, seq, owner=con)
        card.controller, card.location, card.sequence, card.position = con, loc, seq, pos
        if code:
            card.code = code
        card = self._fill_stats(card)
        key = (con, loc)
        if key in self.slots:
            slots = self.slots[key]
            while len(slots) <= seq:
                slots.append(None)
            slots[seq] = card
            return
        if key in self.lists:
            lst = self.lists[key]
            index = min(seq, len(lst))
            skip = old_seq if old_loc == loc else None
            self._shift_chain(con, loc, index, 1, skip)
            lst.insert(index, card)
            for i, c in enumerate(lst):
                c.sequence = i

    def _shift_chain(self, con: int, loc: int, at: int, delta: int, skip_seq: int | None = None) -> None:
        """Move chain links that point into a list at index ≥ `at` (a list shift)."""
        for ch in self.chain:
            if (ch.controller, ch.location) != (con, loc) or ch.sequence < at:
                continue
            if skip_seq is not None and ch.sequence == skip_seq:
                continue
            ch.sequence += delta

    def _move(self, raw: bytes) -> None:
        if len(raw) != 1 + 4 + 10 + 10 + 4:
            return
        (code,) = struct.unpack_from("<I", raw, 1)
        from_con, from_loc, from_seq, from_pos = _loc_info(raw, 5)
        to_con, to_loc, to_seq, to_pos = _loc_info(raw, 15)
        # Resolve the chain handler *before* the move: a list shift can move a different
        # link into (from_con, from_loc, from_seq), which must not be mistaken for the
        # moved card (hidden moves carry code 0).
        hit = self._chain_hit(from_con, from_loc, from_seq, code)
        if from_loc & C.LOCATION_OVERLAY:
            self._detach_overlay(from_con, from_loc, from_seq, from_pos, code)
            if to_loc & C.LOCATION_OVERLAY:
                self._attach_overlay(to_con, to_loc, to_seq, to_pos, code)
            elif to_con in (0, 1):
                # Detached material re-enters a zone (usually the grave): the holder's
                # refreshed record drops it, and no record carries its status.
                self._put_to(None, code, to_con, to_loc, to_seq, to_pos)
            self._track_chain_at(hit, code, to_con, to_loc, to_seq, to_pos)
            return
        if to_loc & C.LOCATION_OVERLAY:
            # Attached as xyz material: the holder is `to_con`, in location
            # `to_loc & ~OVERLAY` at `to_seq`, and `to_pos` is the material index
            # (core `card::get_info_location`). Track it on the holder; no refresh
            # carries the new material list until the holder's zone is next refreshed.
            if from_loc == C.LOCATION_DECK and from_con == self.seat:
                self._deck_remove(from_seq)
            self._take_from(from_con, from_loc, from_seq)
            self._attach_overlay(to_con, to_loc, to_seq, to_pos, code)
            self._track_chain_at(hit, code, to_con, to_loc, to_seq, to_pos)
            return
        if from_con not in (0, 1) or to_con not in (0, 1):
            return
        if from_loc == C.LOCATION_DECK and from_con == self.seat:
            self._deck_remove(from_seq)
        card = self._take_from(from_con, from_loc, from_seq)
        if to_loc == C.LOCATION_DECK and to_con == self.seat:
            self._deck_insert(to_seq, code, card.status if card is not None else 0)
        self._put_to(card, code, to_con, to_loc, to_seq, to_pos)
        self._track_chain_at(hit, code, to_con, to_loc, to_seq, to_pos)

    def _demote_deck(self) -> None:
        """Forget positions but keep the (code, status) pairs known to be in our deck."""
        for seq, code in self.deck_seqs.items():
            if seq in self.deck_stat:
                self.deck_loose[(code, self.deck_stat[seq])] += 1
        self.deck_seqs.clear()
        self.deck_stat.clear()

    def _deck_remove(self, seq: int) -> None:
        """A card left our deck at `seq`: higher known sequences shift down."""
        self.deck_seqs.pop(seq, None)
        self.deck_stat.pop(seq, None)
        self.deck_seqs = {s - 1 if s > seq else s: c for s, c in self.deck_seqs.items()}
        self.deck_stat = {s - 1 if s > seq else s: c for s, c in self.deck_stat.items()}

    def _deck_insert(self, seq: int, code: int, status: int = 0) -> None:
        """A card entered our deck at `seq`: known sequences at/after it shift up."""
        self.deck_seqs = {s + 1 if s >= seq else s: c for s, c in self.deck_seqs.items()}
        self.deck_stat = {s + 1 if s >= seq else s: c for s, c in self.deck_stat.items()}
        if code:
            self.deck_seqs[seq] = code
            if status & STATUS_OBS_MASK:
                self.deck_stat[seq] = status & STATUS_OBS_MASK

    def _chain_hit(self, con: int, loc: int, seq: int, code: int) -> int | None:
        """Index of the chain link whose handler is at (con, loc, seq), if any."""
        for i, ch in enumerate(self.chain):
            if (ch.controller, ch.location, ch.sequence) != (con, loc, seq):
                continue
            if code and ch.code and code != ch.code:
                continue
            return i
        return None

    def _track_chain_at(
        self, index: int | None, code: int, to_con: int, to_loc: int, to_seq: int, to_pos: int
    ) -> None:
        """Chain links name the handler's current position (`query_field` reads it live)."""
        if index is None:
            return
        ch = self.chain[index]
        self.chain[index] = Q.ChainLink(
            ch.code or code,
            to_con,
            to_loc,
            to_seq,
            to_pos,
            ch.triggering_controller,
            ch.triggering_location,
            ch.triggering_sequence,
            ch.description,
        )

    def _entry_at(self, con: int, loc: int, seq: int) -> CardState | None:
        key = (con, loc)
        if key in self.slots and 0 <= seq < len(self.slots[key]):
            return self.slots[key][seq]
        if key in self.lists and 0 <= seq < len(self.lists[key]):
            return self.lists[key][seq]
        return None

    def set_status(self, con: int, loc: int, seq: int, bit: int, on: bool) -> None:
        """Set/clear a status bit on one tracked card (no-op when untracked)."""
        card = self._entry_at(con, loc, seq)
        if card is not None:
            card.status = (card.status | bit) if on else (card.status & ~bit)

    def set_public(self, con: int, loc: int, seq: int, on: bool) -> None:
        """Latch a card's query-IS_PUBLIC (chain relations; no-op when untracked)."""
        card = self._entry_at(con, loc, seq)
        if card is not None:
            card.public = on

    def summon_revert(self, ref: CardRef, db: dict) -> None:
        """A summon was declared for `ref`: the core resets its effects (notably a
        placed monster's EFFECT_CHANGE_TYPE) before the next prompt, with no message.
        Opponent summons need no prompt in that window; ours do (SELECT_POSITION)."""
        card = self._entry_at(ref.controller, ref.location, ref.sequence)
        if card is not None and card.code and card.code in db and card.type != db[card.code].type:
            card.type = db[card.code].type
            self._fill_stats(card, force=True)

    def _attach_overlay(self, con: int, loc: int, seq: int, index: int, code: int) -> None:
        """Attach `code` under the holder at (con, loc & ~OVERLAY, seq), material `index`."""
        holder = self._entry_at(con, loc & ~C.LOCATION_OVERLAY, seq)
        if holder is None:
            return
        overlay = list(holder.overlay)
        while len(overlay) <= index:
            overlay.append(0)
        overlay[index] = code
        holder.overlay = tuple(overlay)

    def _detach_overlay(self, con: int, loc: int, seq: int, index: int, code: int) -> None:
        """Drop material `index` (by code as a fallback) from the holder at (con, loc, seq)."""
        holder = self._entry_at(con, loc & ~C.LOCATION_OVERLAY, seq)
        if holder is None:
            return
        overlay = list(holder.overlay)
        if 0 <= index < len(overlay) and overlay[index] == code:
            del overlay[index]
        elif code in overlay:
            overlay.remove(code)
        else:
            return
        holder.overlay = tuple(overlay)

    def _draw(self, raw: bytes) -> None:
        if len(raw) < 6:
            return
        player, (drawn,) = raw[1], struct.unpack_from("<I", raw, 2)
        if player not in (0, 1) or len(raw) != 6 + 8 * drawn:
            return
        self.deck_count[player] = max(0, self.deck_count[player] - drawn)
        if player == self.seat:
            size = self.deck_count[player]
            self.deck_seqs = {s: c for s, c in self.deck_seqs.items() if s < size}
            self.deck_stat = {s: c for s, c in self.deck_stat.items() if s < size}
        lst = self.lists[(player, C.LOCATION_HAND)]
        for i in range(drawn):
            (code,) = struct.unpack_from("<I", raw, 6 + 8 * i)
            (pos,) = struct.unpack_from("<I", raw, 6 + 8 * i + 4)
            lst.append(CardState(player, C.LOCATION_HAND, len(lst), code, position=pos, owner=player))

    def _chain_link(self, raw: bytes) -> Q.ChainLink:
        (code,) = struct.unpack_from("<I", raw, 1)
        con, loc, seq, pos = _loc_info(raw, 5)
        tcon, tloc = raw[15], raw[16]
        (tseq,) = struct.unpack_from("<I", raw, 17)
        (desc,) = struct.unpack_from("<Q", raw, 21)
        return Q.ChainLink(code, con, loc, seq, pos, tcon, tloc, tseq, desc)

    def _revert_trap_monster(self, ch: Q.ChainLink) -> None:
        """Revert a resolving Continuous-Trap monster to its card-DB type and stats."""
        card = self._entry_at(ch.controller, ch.location, ch.sequence)
        if (
            card is None
            or card.location != C.LOCATION_SZONE
            or not card.position & C.POS_FACEUP
            or card.code not in self.cards_db
            or card.type == self.cards_db[card.code].type
        ):
            return
        card.type = self.cards_db[card.code].type
        self._fill_stats(card, force=True)

    def _pos_change(self, raw: bytes) -> None:
        if len(raw) < 10:
            return
        (code,) = struct.unpack_from("<I", raw, 1)
        cc, cl, cs, _pp, cp = raw[5], raw[6], raw[7], raw[8], raw[9]
        key = (cc, cl)
        card = None
        if key in self.slots and 0 <= cs < len(self.slots[key]):
            card = self.slots[key][cs]
        elif key in self.lists and 0 <= cs < len(self.lists[key]):
            card = self.lists[key][cs]
        if card is not None:
            card.position = cp
            if code:
                card.code = code
            self._fill_stats(card)
        for i, ch in enumerate(self.chain):
            if (ch.controller, ch.location, ch.sequence) == (cc, cl, cs):
                self.chain[i] = Q.ChainLink(
                    code or ch.code,
                    cc,
                    cl,
                    cs,
                    cp,
                    ch.triggering_controller,
                    ch.triggering_location,
                    ch.triggering_sequence,
                    ch.description,
                )

    def _swap(self, raw: bytes) -> None:
        if len(raw) != 1 + 14 + 14:
            return
        (code1,) = struct.unpack_from("<I", raw, 1)
        c1, l1, s1, p1 = _loc_info(raw, 5)
        (code2,) = struct.unpack_from("<I", raw, 15)
        c2, l2, s2, p2 = _loc_info(raw, 19)
        a = self._take_from(c1, l1, s1)
        b = self._take_from(c2, l2, s2)
        self._put_to(a, code1, c2, l2, s2, p2)
        self._put_to(b, code2, c1, l1, s1, p1)

    def _remove_cards(self, raw: bytes) -> None:
        if len(raw) < 6:
            return
        (n,) = struct.unpack_from("<I", raw, 1)
        for i in range(n):
            off = 5 + 10 * i
            if off + 10 > len(raw):
                return
            con, loc, seq, _pos = _loc_info(raw, off)
            self._take_from(con, loc, seq)


def _mzone_code(state: StreamState, con: int, seq: int) -> int:
    for c in state.slots.get((con, C.LOCATION_MZONE), []):
        if c is not None and c.sequence == seq:
            return c.code
    return 0


def history_from_message(state: StreamState, raw: bytes) -> list[HistoryEntry]:
    """Opponent history entries rebuilt from a public message ([] when unmappable): the viewer's opponent's
    `engine.events.public_events` (the engine envs build `obs_history` with the same rule)."""
    events = public_events(raw, state.turn, state.phase, state.turn_player, bool(state.chain),
                           lambda con, seq: _mzone_code(state, con, seq))  # fmt: skip
    return [e for e in events if e.actor != state.seat]


def loc_id(location: int) -> int:
    if location & C.LOCATION_OVERLAY:
        return 8
    return {
        C.LOCATION_DECK: 1,
        C.LOCATION_HAND: 2,
        C.LOCATION_MZONE: 3,
        C.LOCATION_SZONE: 4,
        C.LOCATION_GRAVE: 5,
        C.LOCATION_REMOVED: 6,
        C.LOCATION_EXTRA: 7,
    }.get(location, 0)


class StreamView:
    """A `SeatView` fed by STOC_GAME_MSG payloads (network client)."""

    def __init__(self, deck: list[int] | dict, cards: dict, seat: int = 0) -> None:
        if isinstance(deck, dict):
            main, extra = list(deck["main"]), list(deck.get("extra", ()))
        else:
            main, extra = list(deck), []
        self.state = StreamState(main, extra, cards, seat)
        self._cards = cards
        self._dec: Decision | None = None
        self._opts: list[Option] = []
        self._resp: bytes | None = None
        self._winner: int | None = None
        self._history: list[HistoryEntry] = []
        self._clear_deck_after = False
        self._auto: bytes | None = None  # a singleton response completed with no agent (take_auto pops it)
        self.retries = 0

    # --- SeatView surface ---

    @property
    def seat(self) -> int:
        return self.state.seat

    @property
    def decision(self) -> Decision | None:
        return self._dec

    @property
    def opts(self) -> list[Option]:
        return list(self._opts)

    def obs(self) -> dict:
        if self._dec is None:
            raise RuntimeError("no pending decision")
        board = Board(self.state, self.state.seat)
        return build_obs(board, self._dec, self._opts, self._history, self.state)

    def _side_effects(self, opt: Option) -> None:
        """Core state changes the engine applies when a pick is answered, before any
        message or refresh names them (ygopro-core Processors::{Idle,Battle,Quick}Effect
        / ActivateEffect, MoveToField). Applied to every pick, recorded or not."""
        if opt.card is None:
            return
        if opt.kind in (Kind.SUMMON, Kind.SPSUMMON):
            self.state.summon_revert(opt.card, self._cards)
        if opt.kind == Kind.ACTIVATE:
            self.state.set_status(
                opt.card.controller, opt.card.location, opt.card.sequence, C.STATUS_CHAINING, True
            )

    def take(self, index: int) -> bytes | None:
        """Record our pick and advance; the core response once the decision is complete."""
        if self._dec is None or self._resp is not None:
            raise RuntimeError("no pending decision to take")
        if not 0 <= index < len(self._opts):
            raise IndexError(f"option {index} of {len(self._opts)}")
        opt = self._opts[index]
        self._record(opt)
        self._side_effects(opt)
        self._auto = None  # an agent answered: the response rides this take, not take_auto
        resp = self._dec.choose(index)
        while resp is None:
            # Multi-step decisions: single-option steps take themselves, unrecorded.
            opts = self._dec.options()
            if len(opts) != 1:
                self._opts = opts[:MAX_OPTIONS]
                return None
            self._side_effects(opts[0])
            resp = self._dec.choose(0)
        self._resp = resp
        if self._clear_deck_after:
            # A deck SORT's response reorders the top: drop positions, keep statuses.
            self.state._demote_deck()
            self._clear_deck_after = False
        return resp

    def take_auto(self) -> bytes | None:
        """Next completed singleton response, if any (the `ClientSession` duel loop
        sends these before asking the agent; agent-completed responses ride `take`).
        `_resp` keeps the last response as a record (the parity audit reads it)."""
        if self._auto is not None:
            resp, self._auto = self._auto, None
            return resp
        return None

    @property
    def needs_answer(self) -> bool:
        return self._dec is not None and self._resp is None and self._auto is None

    @property
    def done(self) -> bool:
        return self._winner is not None

    @property
    def winner(self) -> int | None:
        return self._winner

    # --- feed ---

    def feed(self, payload: bytes) -> None:
        """Fold one STOC_GAME_MSG payload in; SELECTs for our seat become pending decisions."""
        if not payload:
            return
        if payload[0] in M.SELECT_TYPES and payload[1] == self.state.seat:
            if self.needs_answer:
                raise RuntimeError("new SELECT while the previous decision is unanswered (desync)")
            msg = M.parse_one(payload)
            self._learn_select(msg)
            self._dec = make_decision(msg, self._cards, lambda: self._known_pool(self.state.seat))
            self._resp = None
            self._auto = None
            self._settle_singletons()
            return
        if payload[0] == C.MSG_WIN:
            self._winner = payload[1]
            return
        if payload[0] == C.MSG_RETRY:
            self.retries += 1
            return
        self.state.feed(payload)
        self._history.extend(history_from_message(self.state, payload))

    def _settle_singletons(self) -> None:
        """Auto-take single-option steps without recording (`auto_single=True` semantics)."""
        assert self._dec is not None
        while True:
            opts = self._dec.options()
            if not opts:
                raise RuntimeError(f"no legal option for {self._dec.msg}")
            if len(opts) > 1:
                self._opts = opts[:MAX_OPTIONS]
                return
            self._side_effects(opts[0])
            resp = self._dec.choose(0)
            if resp is not None:
                self._opts = []
                self._resp = resp
                self._auto = resp  # no agent involved: the session flushes it via take_auto
                if self._clear_deck_after:
                    self.state._demote_deck()
                    self._clear_deck_after = False
                return

    # --- internals ---

    def _learn_select(self, msg: M.Select) -> None:
        """Learn true deck sequences named by our SELECT (search offers)."""
        if isinstance(msg, M.SortCards):
            # The refs name the current order (valid for this decision); the response
            # reorders the deck top, so drop positions (keeping statuses) once done.
            self.state.learn_deck_seqs(msg.cards)
            if any(r.controller == self.state.seat and r.location == C.LOCATION_DECK for r in msg.cards):
                self._clear_deck_after = True
            return
        refs: list = []
        if isinstance(msg, M.SelectCard):
            refs = msg.cards
        elif isinstance(msg, M.SelectTribute):
            refs = [c for c, _ in msg.cards]
        elif isinstance(msg, M.SelectSum):
            refs = [c for c, _ in msg.must] + [c for c, _ in msg.cards]
        elif isinstance(msg, M.SelectUnselectCard):
            refs = msg.selectable + msg.unselectable
        self.state.learn_deck_seqs(refs)

    def _known_pool(self, p: int) -> set[int]:
        """ANNOUNCE_CARD pool: our deck list, codes we can see, public history codes."""
        board = Board(self.state, p)
        codes = set(self.state.deck_list) | {c for c in self.state.extra_list} | board.visible_codes()
        for e in self._history:
            if e.actor == p or e.public:
                codes.add(e.row[2])
        codes.discard(0)
        return codes

    def _record(self, opt: Option) -> None:
        board = Board(self.state, self.state.seat)
        assert self._dec is not None
        self._history.append(
            history_entry(board, opt, self._dec.msg.type, self._dec.step, self.state.turn, self.state.phase)
        )
