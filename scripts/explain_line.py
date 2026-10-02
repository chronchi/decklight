"""Replay a failed line's deepest path and show the board and options where the verifier got stuck.

uv run python scripts/explain_line.py --lines data/demos/lines/salamangreat.jsonl --id salamangreat/01_roadoftheking-vol1:4
"""

import argparse
import json
from pathlib import Path

from ygo_zero.decks import load_ydk
from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.verify import make_config, verify_line
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import Engine
from ygo_zero.engine.resources import CDB_DIR

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
    print(f"{v.reason}: {v.detail}")
    for st in line.steps:
        print(f"  step {st.i}: {st.action} {st.card} | {st.text[:100]}")
    config = make_config(line, deck, args.seed)
    env = CppEnv(engine, config)
    obs = env.reset()
    for i in v.choices:
        obs = env.step(i)
    if obs is None:
        print("deepest path ends the duel")
        return
    cards = obs["cards"][obs["card_mask"]]
    print(
        f"\nafter {len(v.choices)} choices (deepest point), turn {obs['global'][2]}, msg {obs['global'][21]}:"
    )
    for loc in (2, 3, 4, 5, 6):
        mine = [names.get(int(c[0]), int(c[0])) for c in cards if c[3] == 0 and c[1] == loc]
        print(f"  my {LOC[loc]}: {mine}")
    n = int(obs["action_mask"].sum())
    print("  options:")
    for row in obs["actions"][:n]:
        print(f"    {Kind(int(row[0])).name} {names.get(int(row[2]), int(row[2]))}")


if __name__ == "__main__":
    main()
