"""How strong are the bot's turn-1 boards? (primer games, data/primers/summaries.json.gz)

For each game (seat 0 always goes first): what the first player did on turn 1 (summons, activations), the
interruptions it used during the opponent's turn 2 (its chain links on turn 2, split by where the card was: hand =
hand traps, field, GY/banished), how many of the turn player's links it negated, and whether it died on turn 2.
Prints distributions, the turn-2 kill rate by interruption count, and per-deck means.
Usage: uv run python scripts/audit/turn1_boards.py [--min-games 10]
"""

import argparse
import gzip
import json
from collections import Counter, defaultdict

LOC = {1: "deck", 2: "hand", 4: "field", 8: "field", 16: "gy", 32: "banished", 64: "extra"}


def game_row(g):
    assert g["first_player"] == 0, "script assumes seat 0 goes first"
    t2 = [lk for lk in g["links"] if lk["turn"] == 2]
    mine = [lk for lk in t2 if lk["controller"] == 0]
    where = Counter(LOC.get(lk["handler_loc"] & 0x7F, "other") for lk in mine)
    return {
        "deck": g["decks"][0],
        "t1_summons": sum(1 for s in g["summons"] if s["turn"] == 1 and s["seat"] == 0 and s["ok"]),
        "t1_acts": g["turns"][0]["players"][0]["activations"],
        "int": len(mine),
        "int_hand": where["hand"],
        "int_field": where["field"],
        "int_other": len(mine) - where["hand"] - where["field"],
        "negated_opp": sum(1 for lk in t2 if lk["controller"] == 1 and lk["negated"]),
        "opp_links": sum(1 for lk in t2 if lk["controller"] == 1),
        "t2_kill": len(g["turns"]) <= 2 and g["winner"] == 1,
        "won": g["winner"] == 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", default="data/primers/summaries.json.gz")
    ap.add_argument("--min-games", type=int, default=10)
    a = ap.parse_args()
    rows = [game_row(g) for g in json.load(gzip.open(a.summaries))]
    assert rows, "no games"
    n = len(rows)
    mean = lambda rs, k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
    print(f"games {n}; turn-2 kills {sum(r['t2_kill'] for r in rows)} ({mean(rows, 't2_kill'):.3f})")
    for k in ["t1_summons", "t1_acts", "int", "int_hand", "int_field", "int_other", "negated_opp", "opp_links"]:
        print(f"  mean {k:12s} {mean(rows, k):5.2f}")
    print("\ninterruptions used on turn 2 -> games, share, turn-2 kill rate, first-player win rate")
    by = defaultdict(list)
    for r in rows:
        by[min(r["int"], 5)].append(r)
    for k in sorted(by):
        rs = by[k]
        print(f"  {k}{'+' if k == 5 else ' '}  {len(rs):5d}  {len(rs) / n:.3f}  kill {mean(rs, 't2_kill'):.3f}  "
              f"win {mean(rs, 'won'):.3f}")
    print("\nturn-1 summons -> games, mean interruptions, turn-2 kill rate")
    by = defaultdict(list)
    for r in rows:
        by[min(r["t1_summons"], 6)].append(r)
    for k in sorted(by):
        rs = by[k]
        print(f"  {k}{'+' if k == 6 else ' '}  {len(rs):5d}  int {mean(rs, 'int'):.2f}  kill {mean(rs, 't2_kill'):.3f}")
    decks = defaultdict(list)
    for r in rows:
        decks[r["deck"]].append(r)
    ds = [(mean(rs, "int"), d, rs) for d, rs in decks.items() if len(rs) >= a.min_games]
    ds.sort(reverse=True)
    print(f"\nper deck going first (>= {a.min_games} games, {len(ds)} decks): int, hand/field, t1 summons, kill, win")
    for m, d, rs in [*ds[:10], *ds[-10:]]:
        print(f"  {d[:24]:24s} n{len(rs):4d}  int {m:.2f} ({mean(rs, 'int_hand'):.2f}/{mean(rs, 'int_field'):.2f})  "
              f"sum {mean(rs, 't1_summons'):.1f}  kill {mean(rs, 't2_kill'):.2f}  win {mean(rs, 'won'):.2f}")


if __name__ == "__main__":
    main()
