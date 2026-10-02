"""Baseline head-to-head on fixed deck pairs (all ordered pairs of the deck list, both seats).

uv run python scripts/eval_match.py --a greedy --b random --seeds 20 [--decks tests/fixtures/decks]
"""

import argparse
import time
from pathlib import Path

from ygo_zero.decks import load_ydk
from ygo_zero.engine.duel import Engine
from ygo_zero.eval.match import play_match, report
from ygo_zero.eval.policies import GreedyPolicy, RandomPolicy
from ygo_zero.eval.ygoagent import YgoAgentPolicy

POLICIES = {"random": RandomPolicy, "greedy": GreedyPolicy, "ygoagent": YgoAgentPolicy}


def make_policy(name: str, seed: int):
    if name == "ygoagent":  # (model_path, code_list_path, seed=...) — seed is third
        return YgoAgentPolicy(seed=seed)
    return POLICIES[name](seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="greedy", choices=POLICIES)
    ap.add_argument("--b", default="random", choices=POLICIES)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--decks", type=Path, default=Path("tests/fixtures/decks"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    paths = sorted(args.decks.glob("*.ydk"))
    decks = [load_ydk(p) for p in paths]
    pairs = [(i, j) for i in range(len(decks)) for j in range(len(decks)) if i != j]
    t0 = time.time()
    games = play_match(Engine(), decks, make_policy(args.a, 1), make_policy(args.b, 2), pairs,
                       range(args.seeds), num_envs=64, num_threads=args.threads)  # fmt: skip
    print(report(games, [p.stem for p in paths], args.a, args.b))
    print(f"{len(games)} games in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
