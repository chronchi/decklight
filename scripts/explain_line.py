"""Replay a failed line's deepest path and show the board and options where the verifier got stuck.

uv run python scripts/explain_line.py --lines data/demos/lines/salamangreat.jsonl --id salamangreat/01_roadoftheking-vol1:4
"""

import argparse
import json
from pathlib import Path

from decklight.decks import load_ydk
from decklight.demos.explain import explain_failure
from decklight.demos.schema import ComboLine
from decklight.demos.verify import verify_line
from decklight.engine.duel import Engine
from decklight.engine.resources import CDB_DIR

LOC = {1: "deck", 2: "hand", 3: "mzone", 4: "szone", 5: "grave", 6: "banished", 7: "extra", 8: "overlay"}


def card_names() -> dict[int, str]:
    import sqlite3

    con = sqlite3.connect(f"file:{CDB_DIR / 'cards.cdb'}?mode=ro", uri=True)
    return dict(con.execute("SELECT id, name FROM texts").fetchall())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--ydk", default=None, help="default: decks/pool/<deck>.ydk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=2000)
    args = ap.parse_args()
    with open(args.lines) as f:
        raw = next(d for d in map(json.loads, f) if d["id"] == args.id)
    line = ComboLine.from_json(raw)
    deck = load_ydk(args.ydk or Path("decks/pool") / f"{line.deck}.ydk")
    engine, names = Engine(), card_names()
    v = verify_line(engine, line, deck, seed=args.seed, budget=args.budget)
    print(explain_failure(engine, line, deck, v, names))


if __name__ == "__main__":
    main()
