#!/usr/bin/env python3
"""Games between a checkpoint and X (both seats), with behaviour stats; optionally collect decision windows.

    # windows for the weight-editing agents (X vs X): dev = training decks, test = held-out decks
    uv run python -u scripts/weightedit/play.py --a X --pairs 40 --seeds 2 --collect data/cache/weightedit/windows/dev-0.npz
    # an edited checkpoint vs X on held-out decks
    uv run python -u scripts/weightedit/play.py --a data/cache/weightedit/deepseek/edit.pt --holdout --pairs 30 --seeds 2

`--a X` means X itself. Each (deck pair, seed) is played twice, A in seat 0 then seat 1.
Prints i/N progress and a JSON summary (A's and B's behaviour: attack / interrupt rates per window and per turn,
damage per own turn, win rate). Exit 3: 0 games finished or 0 windows collected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from decklight.engine.duel import Engine  # noqa: E402
from decklight.eval.match import wilson  # noqa: E402
from decklight.eval.weightedit import X_CKPT, SeatStats, WindowStore, deck_pairs, load_deck, load_policy, play_game  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="X", help="checkpoint for A (X = X itself)")
    ap.add_argument("--b", default="X", help="checkpoint for B (default X)")
    ap.add_argument("--holdout", action="store_true", help="held-out decks (decks/pool/HOLDOUT) instead of training decks")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="deck-pair sampling seed; game seeds are seed*1000 + k")
    ap.add_argument("--shard", default="0/1", help="i/n: play every n-th game from i")
    ap.add_argument("--collect", type=Path, default=None, help="save windows (A's and B's decisions) to this .npz")
    ap.add_argument("--other-every", type=int, default=8, help="keep 1 in k `other` windows")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON summary here too")
    a = ap.parse_args()
    ckpt = lambda s: X_CKPT if s == "X" else Path(s)  # noqa: E731
    pa, pb = load_policy(ckpt(a.a)), load_policy(ckpt(a.b))
    engine = Engine()
    games = [(da, db, a.seed * 1000 + k) for da, db in deck_pairs(a.holdout, a.pairs, a.seed) for k in range(a.seeds)]
    i, n = map(int, a.shard.split("/"))
    swaps = (0,) if a.a == a.b else (0, 1)  # the same net in both seats plays the same game either way
    jobs = [(da, db, s, swap) for da, db, s in games[i::n] for swap in swaps]  # shard by game, not by seat
    st_a, st_b = SeatStats(), SeatStats()
    store = WindowStore(a.other_every, a.seed) if a.collect else None
    cut, t0 = 0, time.time()
    print(f"{len(jobs)} games (A={a.a}, B={a.b}, {'holdout' if a.holdout else 'train'} decks)", flush=True)
    for j, (da, db, seed, swap) in enumerate(jobs):
        pols, stats = ((pb, pa), (st_b, st_a)) if swap else ((pa, pb), (st_a, st_b))
        r = play_game(engine, pols, (load_deck(da), load_deck(db)), seed, stats=stats, keep=store)
        cut += r["winner"] is None
        if store is not None:
            store.game += 1
        if (j + 1) % 5 == 0 or j + 1 == len(jobs):
            print(f"{j + 1}/{len(jobs)} games, {time.time() - t0:.0f} s, A wins {st_a.wins:.1f}/{st_a.games}", flush=True)
    sa = st_a.summary()
    if sa["games"]:
        lo, hi = wilson(st_a.wins, st_a.games)
        sa["win_ci"] = [round(lo, 3), round(hi, 3)]
    out = {"a": a.a, "b": a.b, "holdout": a.holdout, "cut": cut, "seconds": round(time.time() - t0), "A": sa, "B": st_b.summary()}
    if store is not None:
        out["windows_saved"] = store.save(a.collect)
    print(json.dumps(out, indent=1), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1) + "\n")
    if not st_a.games or (store is not None and not out["windows_saved"]):
        print("ALERT weightedit/play: 0 games or 0 windows", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
