import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "audit"))
import rank_lines as RL  # noqa: E402

CDB = ROOT / "third_party/BabelCDB/cards.cdb"
# (name, field interruption, hand interruption): review findings 2026-11-03 (combo-pool-review)
CASES = [
    ("Infinite Impermanence", True, True),
    ("Ash Blossom & Joyous Spring", True, True),
    ("Effect Veiler", True, True),
    ("Solemn Strike", True, False),
    ("Called by the Grave", True, False),
    ("Call of the Haunted", False, False),
    ("Welcome Labrynth", False, False),
    ("Scapegoat", False, False),
    ("Triple Tactics Talent", False, False),
    ("Pot of Desires", False, False),
    ("Dracotail Horn", True, False),
    ("Dracotail Sting", True, False),
    ("Dracotail Flame", True, False),
    ("Sky Striker Mecha - Shark Cannon", True, False),
    ("Sky Striker Mecha - Widow Anchor", True, False),
    ("Triple Tactics Thrust", False, False),
    ("Rahu Dracotail", False, False),
    ("Bottomless Trap Hole", True, False),
    ("Torrential Tribute", True, False),
    ("Solemn Judgment", True, False),
]


@pytest.fixture(scope="module")
def sets():
    if not CDB.exists():
        pytest.skip("cards.cdb not built")
    return RL.interruption_codes(CDB)


@pytest.mark.parametrize("name,on_field,in_hand", CASES)
def test_interruption_classes(sets, name, on_field, in_hand):
    code = sqlite3.connect(CDB).execute("select id from texts where name = ?", (name,)).fetchone()[0]
    field, hand = sets
    assert (code in field, code in hand) == (on_field, in_hand)
