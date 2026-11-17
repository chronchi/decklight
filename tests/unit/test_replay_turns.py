"""Server-replay helpers and the stream-view fixes found by obs parity (NOTEBOOK 2026-11-10 "obs parity")."""

from __future__ import annotations

import struct

from decklight.engine import constants as C
from decklight.engine.duel import Engine
from decklight.play.replay_turns import _cards, same
from decklight.play.stream import StreamState


def test_card_list_encodings_compare_by_indices():
    bitfield = bytes.fromhex("03000000") + bytes([0b00000110])  # EDOPro client: picks 1 and 2
    u32 = struct.pack("<iI2I", 0, 2, 2, 1)
    u8 = struct.pack("<iI2B", 2, 2, 1, 2)
    assert _cards(bitfield) == _cards(u32) == _cards(u8) == (1, 2)
    assert same(bitfield, u32) and not same(bitfield, struct.pack("<iI2I", 0, 2, 0, 1))


def test_start_seeds_opponent_extra_and_hidden_cards_belong_to_their_controller():
    cards = Engine().cards
    main = [c for c, d in cards.items() if d.type & C.TYPE_MONSTER and not d.type & (C.TYPE_FUSION | C.TYPE_LINK | C.TYPE_XYZ | C.TYPE_SYNCHRO)][:1] * 40
    s = StreamState(main, [], cards, seat=0)
    s.feed(bytes([C.MSG_START, 0]) + struct.pack("<IIHHHH", 8000, 8000, 40, 15, 43, 12))
    assert len(s.query_location(1, C.LOCATION_EXTRA)) == 12
    s.feed(bytes([C.MSG_DRAW, 1]) + struct.pack("<I", 5) + bytes(8 * 5))  # the opponent draws 5 hidden cards
    hand = s.query_location(1, C.LOCATION_HAND)
    assert len(hand) == 5 and all(c.owner == 1 for c in hand)
    assert len(s.query_location(0, C.LOCATION_DECK)) == 40  # their hidden cards are not ours
