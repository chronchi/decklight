"""Where did checkpoint B change the play of checkpoint A? Compare two playtest --save-games runs made on the identical
schedule (same card args/BASE/seeds): games/<match>/<k>/ in both are the same deal, so the rendered decision lists
match line by line until the first decision where the policies chose differently.
Prints per match: games that diverge, outcome flips (A result → B result), the turn and decision type of the first
divergence, and examples of A's pick vs B's pick. Engine-free (reads game.txt).
Usage: python scripts/audit/diff_games.py <A games dir> <B games dir> [--examples N] [--match M]"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def decisions(p: Path) -> tuple[list[str], str]:
    txt = p.read_text().splitlines()
    res = "WIN" if " — WIN" in txt[0] else ("LOSS" if " — LOSS" in txt[0] else "DRAW")
    i = next((j for j, ln in enumerate(txt) if ln.startswith("--- decisions")), len(txt))
    return [re.sub(r"^\[\d+\] ", "", ln) for ln in txt[i + 1:] if ln.startswith("[")], res


def deal_index(match_dir: Path) -> dict:
    """(decks, seed, policy seat) -> game dir: saved-game numbering differs between runs, the deal does not."""
    out = {}
    for g in match_dir.iterdir():
        sj = g / "state.json"
        if sj.is_file():
            s = json.loads(sj.read_text())
            out[(tuple(s["decks"]), s["seed"], s["meta"]["policy_seat"])] = g
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--match", default=None)
    args = ap.parse_args()
    for md in sorted(p for p in args.a.iterdir() if p.is_dir()):
        if args.match and md.name != args.match:
            continue
        flips, kinds, turns, ex, n, div = Counter(), Counter(), Counter(), [], 0, 0
        ia, ib = deal_index(md), deal_index(args.b / md.name)
        for key in sorted(ia.keys() & ib.keys()):
            fa, fb = ia[key] / "game.txt", ib[key] / "game.txt"
            if not (fa.is_file() and fb.is_file()):
                continue
            n += 1
            da, ra = decisions(fa)
            db, rb = decisions(fb)
            flips[f"{ra}->{rb}"] += 1
            i = next((j for j, (x, y) in enumerate(zip(da, db)) if x != y), None)
            if i is None:
                continue
            div += 1
            m = re.match(r"T(\d+) P\d (\S+)", da[i])
            if m:
                turns[min(int(m.group(1)), 6)] += 1
                kinds[m.group(2)] += 1
            if len(ex) < args.examples and ra != rb:
                ex.append(f"  {Path(key[0][0]).stem} vs {Path(key[0][1]).stem} P{key[2]} {ra}->{rb} at decision {i}:\n    A: {da[i]}\n    B: {db[i]}")
        if n == 0:
            continue
        print(f"== {md.name}: {n} games, {div} diverge ({div / n:.0%})")
        print("  outcome A->B: " + ", ".join(f"{k} {v}" for k, v in sorted(flips.items())))
        print("  first divergence turn (6 = 6+): " + ", ".join(f"T{k}:{v}" for k, v in sorted(turns.items())))
        print("  first divergence kind: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
        print("\n".join(ex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
