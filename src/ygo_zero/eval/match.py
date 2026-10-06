"""Head-to-head matches on fixed deck pairs with seat swapping, and win-rate reports with confidence intervals.

Every (deck_a, deck_b, seed) is played twice: A going first with deck_a, and A going second with deck_a (same
seed), so seat and deck advantages cancel. Player 0 always moves first in the core.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

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
    winner_is_a: float  # 1 win, 0 loss, 0.5 draw (or cut at max_decisions)
    turns: int
    truncated: bool = False
    choices: tuple = field(default=(), compare=False)  # response indices, replayable via CppEnv


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ring_pairs(n: int, k: int) -> list[tuple[int, int]]:
    """Each of n decks vs its next k decks (cyclic): n*k ordered pairs instead of n*(n-1), every deck equally often
    on both sides. k >= n-1 gives all pairs."""
    k = min(k, n - 1)
    return [(i, (i + d) % n) for i in range(n) for d in range(1, k + 1)]


def play_match(
    engine: Engine,
    decks: Sequence,
    a: Policy,
    b: Policy,
    pairs: Sequence[tuple[int, int]],
    seeds: Sequence[int],
    num_envs: int = 32,
    num_threads: int = 8,
    max_decisions: int = 3000,
    mode: str = "async",
    recv_batch: int | None = None,
) -> list[Game]:
    """`max_decisions`: a game still running after that many decisions (both seats) is cut and scored as a draw:
    two deterministic policies can loop for tens of thousands of decisions, and one such game stalls the whole
    match (RL smoke and playtests, 2026-10-05). 0 = no cap.
    `mode`: "sync" steps every env in lockstep (one barrier per decision); "async" collects ready envs via
    `recv_batch` (default `max(1, num_envs // 4)`) so slow engine steps don't stall the rest."""
    if mode not in ("sync", "async"):
        raise ValueError(f"mode must be 'sync' or 'async', got {mode!r}")
    schedule, meta = [], []
    for da, db in pairs:
        for s in seeds:
            schedule.append((da, db, s))  # A is player 0 (first)
            meta.append((da, db, s, True))
            schedule.append((db, da, s))  # A is player 1 (second)
            meta.append((da, db, s, False))
    vec = VecEnv(engine, decks, min(num_envs, len(schedule)), num_threads, seed=0)
    vec.set_schedule(schedule)
    vec.set_max_decisions(max_decisions)
    seat_of_a = np.array([0 if m[3] else 1 for m in meta] + [-2], np.int32)  # [-1] → -2: idle env
    games: list[Game] = []

    if mode == "sync":
        buf = vec.reset()
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
                win = 0.5 if ep.winner not in (0, 1) else float(ep.winner == seat)
                games.append(Game(da, db, s, a_first, win, ep.turns, ep.winner == VecEnv.TRUNCATED,
                                  tuple(int(c) for c in ep.choices)))
    else:
        buf = vec.async_reset()
        n = vec.num_envs
        batch = recv_batch if recv_batch is not None else max(1, n // 4)
        while True:
            ids = np.asarray(vec.recv(batch, n), dtype=np.int64)
            for ep in vec.pop_finished():
                da, db, s, a_first = meta[ep.index]
                seat = 0 if a_first else 1
                win = 0.5 if ep.winner not in (0, 1) else float(ep.winner == seat)
                games.append(Game(da, db, s, a_first, win, ep.turns, ep.winner == VecEnv.TRUNCATED,
                                  tuple(int(c) for c in ep.choices)))
            if len(ids) == 0:  # nothing in flight: every env received and idle
                if vec.remaining != 0:
                    raise RuntimeError(
                        f"play_match async stalled: recv empty with {vec.remaining} games remaining"
                    )
                break
            live = np.ascontiguousarray(ids[buf["player"][ids] != -1])  # idle rows are never sent again
            if len(live) == 0:
                if vec.remaining == 0:
                    break
                continue
            a_seat = seat_of_a[buf["game"][live]]
            player = buf["player"][live]
            mine_pos = np.flatnonzero(player == a_seat)
            theirs_pos = np.flatnonzero(player != a_seat)
            send_ids = np.ascontiguousarray(live)
            send_acts = np.zeros(len(live), np.int32)
            if len(mine_pos):
                send_acts[mine_pos] = a(buf, np.ascontiguousarray(live[mine_pos]))
            if len(theirs_pos):
                send_acts[theirs_pos] = b(buf, np.ascontiguousarray(live[theirs_pos]))
            vec.send(send_ids, send_acts)
        for ep in vec.pop_finished():  # games that finished in the last in-flight batch
            da, db, s, a_first = meta[ep.index]
            seat = 0 if a_first else 1
            win = 0.5 if ep.winner not in (0, 1) else float(ep.winner == seat)
            games.append(Game(da, db, s, a_first, win, ep.turns, ep.winner == VecEnv.TRUNCATED,
                              tuple(int(c) for c in ep.choices)))
    games.sort(key=lambda g: (g.deck_a, g.deck_b, g.seed, g.a_first))
    return games


ROOT = Path(__file__).resolve().parents[3]


def game_state(
    deck_paths: Sequence[str | Path],
    game: Game,
    engine: Engine,
    *,
    match: str = "",
    opponent: str = "",
) -> dict:
    """A finished `Game` as a hand_play state dict (replay: decks + seed + choice list).

    `deck_paths` is the deck list `play_match` ran on, as `.ydk` paths relative to the repo
    root (seat order is resolved here: seat 0 first). `actors`/`turns` are recomputed by
    replaying the choices through `CppEnv` (engine only, no model). Every decision is
    labelled "gold".
    """
    from ygo_zero.decks import load_ydk
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.engine.duel import DuelConfig

    seat0_idx = game.deck_a if game.a_first else game.deck_b
    seat1_idx = game.deck_b if game.a_first else game.deck_a
    paths = [str(p).replace("\\", "/") for p in (deck_paths[seat0_idx], deck_paths[seat1_idx])]

    def _load(p: str):
        q = Path(p)
        if not q.is_absolute():
            q = ROOT / p
        return load_ydk(q)

    cfg = DuelConfig(_load(paths[0]), _load(paths[1]), seed=game.seed)
    env = CppEnv(engine, cfg)
    obs = env.reset()
    actors, turns = [], []
    for c in game.choices:
        actors.append(int(env.player))
        turns.append(int(obs["global"][2]))
        obs = env.step(int(c))
        if obs is None:
            break
    policy_seat = 0 if game.a_first else 1
    if game.truncated or game.winner_is_a == 0.5:
        winner = None
    elif game.winner_is_a == 1.0:
        winner = policy_seat
    else:
        winner = 1 - policy_seat
    return {
        "decks": paths,
        "seed": game.seed,
        "choices": [int(c) for c in game.choices],
        "labels": ["gold"] * len(game.choices),
        "actors": actors,
        "turns": turns,
        "notes": [],
        "meta": {
            "match": match,
            "policy_seat": policy_seat,
            "opponent": opponent,
            "winner": winner,
            "policy_won": bool(game.winner_is_a == 1.0),
            "turns": game.turns,
            "truncated": bool(game.truncated),
        },
    }


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
    out.append(f"mean turns {np.mean([g.turns for g in games]):.1f}; cut at the decision cap: {sum(g.truncated for g in games)}")
    return "\n".join(out)
