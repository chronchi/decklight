"""Read verified going-first lines as text, to judge them directly (not by end-board count): hand, every decision of
player 0, end board; plus whether the line's deck is the pool deck it is filed under.

    uv run python scripts/audit/show_verified.py azamina-white-forest-meta1,branded --n 2 --seed 1

Laptop-safe: engine replay only, no model.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

import playtest as PT  # noqa: E402
import show_lines as SL  # noqa: E402

from decklight.decks import load_ydk  # noqa: E402
from decklight.demos.names import CardNames  # noqa: E402
from decklight.engine.duel import Engine  # noqa: E402


def deck_match(rec: dict, name: str) -> str:
    """How far the line's main deck is from the pool deck (cards in one but not the other)."""
    pool = Counter(load_ydk(ROOT / "decks/pool" / f"{name}.ydk").main)
    line = Counter(rec["config"]["deck0_main"])
    hand = len(rec["config"].get("setup", []))  # the opening hand is placed by setup, not drawn from deck0_main
    extra, missing = sum((line - pool).values()), sum((pool - line).values())
    if extra == 0 and missing == hand:
        return "pool deck"
    return f"DECK DIFFERS from decks/pool/{name}.ydk: +{extra} cards, -{missing - hand} beyond the {hand}-card hand"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("decks")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng, engine, nn = random.Random(a.seed), Engine(), CardNames().names
    for name in a.decks.split(","):
        recs = PT.verified_records(name, limit=10_000)
        if not recs:
            print(f"== {name}: no verified lines")
            continue
        print(f"== {name}: {len(recs)} lines, sources {dict(Counter(r.get('deck_source', '?') for r in recs))}")
        for rec in rng.sample(recs, min(a.n, len(recs))):
            print(f"  -- {rec['line_id']} ({deck_match(rec, name)}; inferred {rec.get('inferred_frac', 0):.0%})")
            row = dict(rec, field=len(rec.get("end_board_codes", [])), v_p0=0.0, stuck=False)
            print("\n".join(SL.render(row, engine, nn)))
            print(f"    record end_board: {', '.join(rec.get('end_board', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
