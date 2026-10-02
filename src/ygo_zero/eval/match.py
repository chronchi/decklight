"""Head-to-head matches on fixed deck pairs with seat swapping, and win-rate reports with confidence intervals.

Every (deck_a, deck_b, seed) is played twice: A going first with deck_a, and A going second with deck_a (same
seed), so seat and deck advantages cancel. Player 0 always moves first in the core.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ygo_zero.engine.cpp_env import VecEnv
from ygo_zero.engine.duel import Engine
from ygo_zero.eval.policies import Policy


@dataclass(frozen=True)
class Game:
    deck_a: int  # index into the deck list, played by A
    deck_b: int
    seed: int
    a_first: bool
    winner_is_a: float  # 1 win, 0 loss, 0.5 draw
    turns: int


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def play_match(
    engine: Engine,
    decks: Sequence,
    a: Policy,
    b: Policy,
    pairs: Sequence[tuple[int, int]],
    seeds: Sequence[int],
    num_envs: int = 32,
    num_threads: int = 8,
) -> list[Game]:
    schedule, meta = [], []
    for da, db in pairs:
        for s in seeds:
            schedule.append((da, db, s))  # A is player 0 (first)
            meta.append((da, db, s, True))
            schedule.append((db, da, s))  # A is player 1 (second)
            meta.append((da, db, s, False))
    vec = VecEnv(engine, decks, min(num_envs, len(schedule)), num_threads, seed=0)
    vec.set_schedule(schedule)
    buf = vec.reset()
    seat_of_a = np.array([0 if m[3] else 1 for m in meta] + [-2], np.int32)  # [-1] → -2: idle env
    games: list[Game] = []

    while vec.remaining > 0:
        player = buf["player"]
        a_seat = seat_of_a[buf["game"]]
        actions = np.zeros(vec.num_envs, np.int32)
        live = player >= 0
        mine = np.flatnonzero(live & (player == a_seat))
        theirs = np.flatnonzero(live & (player != a_seat))
        if len(mine):
            actions[mine] = a(buf, mine)
        if len(theirs):
            actions[theirs] = b(buf, theirs)
        vec.step(actions)
        for ep in vec.pop_finished():
            da, db, s, a_first = meta[ep.index]
            seat = 0 if a_first else 1
            win = 0.5 if ep.winner == 2 else float(ep.winner == seat)
            games.append(Game(da, db, s, a_first, win, ep.turns))
    return games


def report(games: Sequence[Game], deck_names: Sequence[str], a_name: str, b_name: str) -> str:
    def line(label, gs):
        w = sum(g.winner_is_a for g in gs)
        lo, hi = wilson(w, len(gs))
        return f"{label:<40} {w / len(gs):6.3f}  [{lo:.3f}, {hi:.3f}]  n={len(gs)}"

    out = [f"{a_name} vs {b_name}: A win rate [95% Wilson CI]", line("overall", games)]
    out.append(line("A going first", [g for g in games if g.a_first]))
    out.append(line("A going second", [g for g in games if not g.a_first]))
    by_pair = defaultdict(list)
    for g in games:
        by_pair[(g.deck_a, g.deck_b)].append(g)
    for (da, db), gs in sorted(by_pair.items()):
        out.append(line(f"  {deck_names[da]} vs {deck_names[db]}", gs))
    out.append(f"mean turns {np.mean([g.turns for g in games]):.1f}")
    return "\n".join(out)
