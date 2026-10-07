"""Honest-world benchmark: engine only, random policy on pool decks.

At 5 decision points per game (turn >= 2), draws 4 worlds each; prints the acceptance rate,
fallback rate, ms per proposal (replay), and the fraction of accepted worlds where the
opponent's hidden hand codes (resp. my next draw) differ from the truth.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.decks import load_ydk  # noqa: E402
from ygo_zero.engine import constants as C  # noqa: E402
from ygo_zero.engine.cpp_env import CppEnv  # noqa: E402
from ygo_zero.engine.duel import DuelConfig, Engine  # noqa: E402
from ygo_zero.search.determinize import Determinizer, SeenTracker  # noqa: E402


def _deck_top(cards: dict, team: int) -> int | None:
    """My next draw: the code of the deck card with the highest sequence (draws pop the top)."""
    m = (cards["player"] == team) & (cards["location"] == C.LOCATION_DECK)
    if not m.sum():
        return None
    seq = cards["sequence"][m]
    return int(cards["code"][m][seq.argmax()])


def _hidden_hand(cards: dict, opp: int) -> list[int]:
    return sorted(
        int(cards["code"][i]) for i in range(len(cards["code"]))
        if int(cards["player"][i]) == opp and int(cards["location"][i]) == C.LOCATION_HAND
        and not bool(cards["known"][i])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--points", type=int, default=5)
    args = ap.parse_args()

    engine = Engine()
    pool = sorted((ROOT / "decks" / "pool").glob("*.ydk"))
    assert len(pool) >= 2, "no pool decks"
    rng = random.Random(args.seed)

    proposals = accepted = fallbacks = 0
    seconds = 0.0
    n_hand = n_hand_diff = 0
    n_draw = n_draw_diff = 0
    n_pts = 0
    t_all = time.perf_counter()
    for g in range(args.games):
        d0, d1 = rng.sample(pool, 2)
        cfg = DuelConfig(load_ydk(d0), load_ydk(d1), seed=args.seed * 1_000 + g)
        env = CppEnv(engine, cfg, True)
        obs = env.reset()
        t = {0: SeenTracker(0), 1: SeenTracker(1)}
        t[0].update(env)
        t[1].update(env)
        pts: list[tuple[list[int], int, dict, set[int]]] = []
        while obs is not None and len(env.choices) < 2000:
            if int(obs["global"][2]) >= 2:
                pts.append((list(env.choices), int(env.player), obs, set(t[int(env.player)].seen)))
            legal = np.flatnonzero(obs["action_mask"]).tolist()
            obs = env.step(rng.choice(legal))
            t[0].update(env)
            t[1].update(env)
        if len(pts) < args.points:
            print(f"game {g}: only {len(pts)} turn>=2 points, skipped")
            continue
        for prefix, me, root_obs, seen in rng.sample(pts, args.points):
            det = Determinizer(engine, cfg, True, me, prefix, seen, root_obs,
                               random.Random(f"bench:{g}:{len(prefix)}"), tries=8)
            for _ in range(args.worlds):
                world = det.sample()
                if world is None:
                    continue
                # opponent hidden hand vs truth
                env_t, env_w = CppEnv(engine, cfg, True), CppEnv(engine, world, True)
                env_t.reset()
                env_w.reset()
                for c in prefix:
                    env_t.step(c)
                    env_w.step(c)
                ct, cw = env_t.cards(), env_w.cards()
                opp = 1 - me
                n_hand += 1
                n_hand_diff += _hidden_hand(cw, opp) != _hidden_hand(ct, opp)
                dt, dw = _deck_top(ct, me), _deck_top(cw, me)
                if dt is not None and dw is not None:
                    n_draw += 1
                    n_draw_diff += dt != dw
            proposals += det.proposals
            accepted += det.accepted
            fallbacks += det.fallbacks
            seconds += det.seconds
            n_pts += 1
    wall = time.perf_counter() - t_all
    worlds = accepted + fallbacks
    print(f"points: {n_pts}, acceptance {accepted}/{proposals} = {accepted / proposals if proposals else 0:.3f}")
    print(f"fallback rate {fallbacks}/{worlds} = {fallbacks / worlds if worlds else 0:.3f}")
    print(f"ms per proposal (replay): {1000 * seconds / proposals if proposals else 0:.1f}")
    print(f"opp hidden hand differs: {n_hand_diff}/{n_hand} = {n_hand_diff / n_hand if n_hand else 0:.3f}")
    print(f"my next draw differs: {n_draw_diff}/{n_draw} = {n_draw_diff / n_draw if n_draw else 0:.3f}")
    print(f"wall {wall:.0f} s")


if __name__ == "__main__":
    main()
