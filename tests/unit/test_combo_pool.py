import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "audit"))
import combo_pool as CP  # noqa: E402


def _args(**kw):
    return argparse.Namespace(**{"decks": "", "all": True, "shard": "0/1", **kw})


def test_all_candidates_include_pinned_decks_and_shards_partition():
    names = CP.candidates(_args())
    assert set(CP.PINNED_DECKS) <= set(names)
    assert len(names) > 100
    shards = [CP.candidates(_args(shard=f"{i}/3")) for i in range(3)]
    assert sorted(sum(shards, [])) == sorted(names)


def test_unknown_deck_fails_loud():
    with pytest.raises(SystemExit, match="no such pool deck"):
        CP.candidates(_args(decks="no-such-deck-xyz"))


def test_turn_lp_reads_damage_recover_cost_per_turn():
    C = CP.C

    def msg(t, player, amount):
        return bytes([t, player]) + amount.to_bytes(4, "little")

    trace = [bytes([C.MSG_NEW_TURN, 0]), msg(C.MSG_PAY_LPCOST, 0, 1000), msg(C.MSG_DAMAGE, 1, 500),
             bytes([C.MSG_NEW_TURN, 1]), msg(C.MSG_DAMAGE, 0, 2500), msg(C.MSG_DAMAGE, 0, 3000),
             msg(C.MSG_RECOVER, 0, 800)]  # fmt: skip
    per = CP.turn_lp(trace, 0)
    assert per == {1: {"damage": 0, "recover": 0, "cost": 1000}, 2: {"damage": 5500, "recover": 800, "cost": 0}}
    assert CP.turn_lp(trace, 1) == {1: {"damage": 500, "recover": 0, "cost": 0}}
