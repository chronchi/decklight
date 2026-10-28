"""Card-knowledge memory: what each player knows about the other's hidden hand cards.

`HandMemory` keeps, per player, the hand sequence aligned with the core's: `known[p][i]` is
True when player 1-p knows that card's identity (a searched card that was revealed, a monster
bounced from the field, a public draw, ...), and `codes[p][i]` is the true passcode (needed
for the shuffle realignment, which only sees codes).

Updated from the core's informational messages in `Duel.advance` (both the Python reference
duel and the C++ duel, in lockstep). Layouts verified against the core's message writers in
`third_party/ygopro-core/`:

- MSG_MOVE (0x32, 29 bytes): code u32 + from loc_info (con u8, loc u8, seq u32, pos u32) +
  to loc_info + reason u32. Written before the removal / after the insertion, so `from.seq`
  is the pre-removal index and `to.seq` the post-insertion index. Hand adds always append
  (`field::add_card`: push_back); removals erase + resequence (`remove_card` + `reset_sequence`).
- MSG_DRAW (0x5A): player u8 + drawn u32 + drawn x (code u32, position u32), in hand-append
  order. The opening hands arrive this way too (Processors::Startup draws via Processors::Draw).
- MSG_CONFIRM_CARDS (0x1F, ConfirmCards in libduel.cpp): viewer u8 + count u32 + count x
  (code u32, controller u8, location u8, sequence u32). `viewer` is the player the cards are
  shown to; confirm-to-self does not count.
- MSG_SHUFFLE_HAND (0x21): player u8 + count u32 + count x code u32, in the new hand order.
  Identical copies are interchangeable, so the known flags are re-aligned by code. The Rust core
  (third_party/ygopro-core-rust, "Differences from upstream") also sets the top bit of a face-up
  card's code and appends count x u8 (each card's index before the shuffle); both are ignored
  here, so the two cores give the same knowledge.
- MSG_SWAP_GRAVE_DECK never touches hands; MSG_TAG_SWAP only exists in tag duels (1v1: never
  sent). Anything else that would desynchronise the lists is caught by the resync guard in
  `Duel.advance` (tracked length vs queried hand size at every decision).
"""

import struct
from collections import Counter

from decklight.engine import constants as C

MOVE_LEN = 1 + 4 + 10 + 10 + 4  # MSG_MOVE is always code + from + to + reason


def _loc_info(raw: bytes, off: int) -> tuple[int, int, int, int]:
    """(controller, location, sequence, position) at offset `off`."""
    con, loc = raw[off], raw[off + 1]
    (seq,) = struct.unpack_from("<I", raw, off + 2)
    (pos,) = struct.unpack_from("<I", raw, off + 6)
    return con, loc, seq, pos


def source_public(from_loc: int, from_pos: int) -> bool:
    """Was a card leaving (from_loc, from_pos) public to a player other than its holder?

    Mirrors `Board._public` (obs.py): face-up field, GY, face-up banished / face-up Extra Deck,
    and overlay materials (always public) count. Deck / face-down / hand (handled by the caller)
    do not.
    """
    if from_loc & C.LOCATION_OVERLAY:
        return True
    if from_loc == C.LOCATION_GRAVE:
        return True
    if from_loc in (C.LOCATION_REMOVED, C.LOCATION_EXTRA):
        return bool(from_pos & C.POS_FACEUP)
    if from_loc in (C.LOCATION_MZONE, C.LOCATION_SZONE):
        return bool(from_pos & C.POS_FACEUP)
    return False


class HandMemory:
    """Per-player hand knowledge + true codes, both aligned with the core's hand sequences."""

    def __init__(self, setup_hands: tuple[list[int], list[int]] = ([], [])) -> None:
        # Cards placed in hands before the duel starts (puzzle setup): no messages announce them,
        # so they seed the lists (unknown to the opponent). Everything else arrives via messages.
        self.known: list[list[bool]] = [[False] * len(h) for h in setup_hands]
        self.codes: list[list[int]] = [[int(c) for c in h] for h in setup_hands]

    def note(self, raw: bytes) -> bool:
        """Fold one raw core message in. Returns False when a shuffle length mismatch rebuilt
        a list (the caller counts the resync); True otherwise."""
        if not raw:
            return True
        t = raw[0]
        if t == C.MSG_MOVE:
            self._move(raw)
        elif t == C.MSG_DRAW:
            self._draw(raw)
        elif t == C.MSG_CONFIRM_CARDS:
            self._confirm(raw)
        elif t == C.MSG_SHUFFLE_HAND:
            return self._shuffle(raw)
        return True

    def guard(self, sizes: tuple[int, int]) -> bool:
        """Resync guard: rebuild any list whose length differs from the queried hand size as
        all-unknown. Returns False when anything was rebuilt (the caller counts the resync)."""
        ok = True
        for p in (0, 1):
            if len(self.known[p]) != sizes[p]:
                self.known[p] = [False] * sizes[p]
                self.codes[p] = [0] * sizes[p]
                ok = False
        return ok

    # --- messages ---

    def _move(self, raw: bytes) -> None:
        if len(raw) != MOVE_LEN:
            return
        (code,) = struct.unpack_from("<I", raw, 1)
        from_con, from_loc, from_seq, from_pos = _loc_info(raw, 5)
        to_con, to_loc, to_seq, to_pos = _loc_info(raw, 15)
        from_hand = from_loc == C.LOCATION_HAND and from_con in (0, 1)
        to_hand = to_loc == C.LOCATION_HAND and to_con in (0, 1)
        if not from_hand and not to_hand:
            return
        old = False
        if from_hand and 0 <= from_seq < len(self.known[from_con]):
            old = self.known[from_con][from_seq]
            del self.known[from_con][from_seq]
            del self.codes[from_con][from_seq]
        if to_hand:
            if from_hand and from_con == to_con:
                known = old  # same-player reorder (defensive; move_card refuses these)
            elif from_hand:
                known = True  # a card changing hands: the previous holder knows what they gave
            else:
                known = bool(to_pos & C.POS_FACEUP) or source_public(from_loc, from_pos)
            self.known[to_con].insert(min(to_seq, len(self.known[to_con])), known)
            self.codes[to_con].insert(min(to_seq, len(self.codes[to_con])), code)

    def _draw(self, raw: bytes) -> None:
        if len(raw) < 6:
            return
        player, (drawn,) = raw[1], struct.unpack_from("<I", raw, 2)
        if player not in (0, 1) or len(raw) != 6 + 8 * drawn:
            return
        for i in range(drawn):
            (code,) = struct.unpack_from("<I", raw, 6 + 8 * i)
            (pos,) = struct.unpack_from("<I", raw, 6 + 8 * i + 4)
            self.known[player].append(bool(pos & C.POS_FACEUP))
            self.codes[player].append(code)

    def _confirm(self, raw: bytes) -> None:
        if len(raw) < 6:
            return
        viewer, (count,) = raw[1], struct.unpack_from("<I", raw, 2)
        if viewer not in (0, 1) or len(raw) != 6 + 10 * count:
            return
        for i in range(count):
            off = 6 + 10 * i
            con, loc = raw[off + 4], raw[off + 5]
            (seq,) = struct.unpack_from("<I", raw, off + 6)
            if loc == C.LOCATION_HAND and con in (0, 1) and viewer != con and 0 <= seq < len(
                self.known[con]
            ):
                self.known[con][seq] = True

    def _shuffle(self, raw: bytes) -> bool:
        if len(raw) < 6:
            return True
        player, (count,) = raw[1], struct.unpack_from("<I", raw, 2)
        if player not in (0, 1) or len(raw) not in (6 + 4 * count, 6 + 5 * count):
            return True
        codes = [struct.unpack_from("<I", raw, 6 + 4 * i)[0] & 0x7FFFFFFF for i in range(count)]
        cur, cur_codes = self.known[player], self.codes[player]
        if len(codes) != len(cur) or len(cur_codes) != len(cur):
            self.known[player] = [False] * len(codes)
            self.codes[player] = list(codes)
            return False
        keep: Counter[int] = Counter()
        for known, code in zip(cur, cur_codes, strict=True):
            if known:
                keep[code] += 1
        out: list[bool] = []
        for c in codes:
            out.append(keep[c] > 0)
            if keep[c]:
                keep[c] -= 1
        self.known[player] = out
        self.codes[player] = list(codes)
        return True
