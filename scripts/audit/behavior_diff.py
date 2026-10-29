"""What does policy B do differently from policy A over whole games? Two raw search_match arms on the same deals.

Replays every game of each arm and labels each of OUR seat's decisions "<kind> <card> <effect text>" (as
show_divergence does). PLACE / POSITION decisions are skipped (zone and battle position choices). Per label it
reports, for A and B, how many games use it at least once (share of games), and the win rate of those games. It
prints the labels whose share changed most from A to B, both directions.
Engine replay only (laptop-safe, no model).

    python scripts/audit/behavior_diff.py runs/<job>/card/P-invoked runs/<job>/card/D-invoked \\
        [--deck-dir data/x4b_flat] [--top 15] [--min-games 20] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import show_divergence as SD

SKIP = {"PLACE", "POSITION"}


def labels_of(g: dict, engine, nn: dict) -> set[str]:
    from decklight.demos.game import _opts
    from decklight.engine.cpp_env import CppEnv

    env = CppEnv(engine, SD._config(g), True)
    obs = env.reset()
    out = set()
    for c, actor in zip(g["choices"], g["actors"], strict=False):
        if actor == g["s_seat"]:
            kind = SD.H._kind(_opts(obs)[int(c)][0])
            if kind not in SKIP:
                out.add(SD._label(obs, int(c), nn))
        obs = env.step(int(c))
    return out


def arm_stats(d: str, engine, nn: dict) -> tuple[int, Counter, Counter]:
    games = SD.load(d)
    if not games:
        raise SystemExit(f"no games in {d}")
    use, win = Counter(), Counter()
    for g in games.values():
        w = SD.res(g) == "W"
        for lab in labels_of(g, engine, nn):
            use[lab] += 1
            win[lab] += w
    return len(games), use, win


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--deck-dir", type=Path, default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    from decklight.demos.names import CardNames
    from decklight.engine.duel import Engine

    if a.deck_dir is not None:
        SD.DECK_DIRS.insert(0, a.deck_dir)
    nn, engine = CardNames().names, Engine()
    na, ua, wa = arm_stats(a.a, engine, nn)
    nb, ub, wb = arm_stats(a.b, engine, nn)
    rows = []
    for lab in set(ua) | set(ub):
        if ua[lab] + ub[lab] < a.min_games:
            continue
        rows.append({"label": lab, "A_share": ua[lab] / na, "B_share": ub[lab] / nb,
                     "A_wr": wa[lab] / ua[lab] if ua[lab] else None, "B_wr": wb[lab] / ub[lab] if ub[lab] else None,
                     "A_n": ua[lab], "B_n": ub[lab]})
    rows.sort(key=lambda r: r["B_share"] - r["A_share"])

    def show(r: dict) -> str:
        f = lambda x: "  -  " if x is None else f"{x:.2f}"
        return (f"  {r['A_share']:5.1%} -> {r['B_share']:5.1%} ({(r['B_share'] - r['A_share']) * 100:+5.1f} pp)  "
                f"wr {f(r['A_wr'])}/{f(r['B_wr'])}  {r['label'][:110]}")

    print(f"A {a.a}: {na} games; B {a.b}: {nb} games. Share of games using the play at least once (A -> B), "
          "win rate of those games (A/B)")
    print(f"B uses MORE (top {a.top}):")
    for r in reversed(rows[-a.top:]):
        print(show(r))
    print(f"B uses LESS (top {a.top}):")
    for r in rows[: a.top]:
        print(show(r))
    if a.json:
        a.json.write_text(json.dumps({"A": a.a, "B": a.b, "nA": na, "nB": nb, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
