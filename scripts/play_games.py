"""Play full games between pool decks with the turn planner + engine verifier (ADR 0004, src/ygo_zero/demos/game.py).

nice -n 19 uv run python scripts/play_games.py --pairs blue-eyes:sky-striker --games 1 --max-turns 4
Records: data/demos/games/<deck0>-vs-<deck1>.jsonl (append), one game per line.
"""

import argparse
import json
from pathlib import Path

from ygo_zero.decks import load_ydk
from ygo_zero.demos import llm as _llm
from ygo_zero.demos.game import TURN_SYSTEM, play_game
from ygo_zero.demos.names import CardNames
from ygo_zero.demos.schema import ComboLine
from ygo_zero.engine.duel import Engine


def verified_examples(deck: str, k: int = 2) -> list[ComboLine]:
    """A few of this deck's verified Claude lines as few-shot examples."""
    ver = Path(f"data/demos/claude/verified/{deck}.jsonl")
    lines = Path(f"data/demos/claude/lines/{deck}.jsonl")
    if not ver.exists() or not lines.exists():
        return []
    ok = [json.loads(x)["line_id"] for x in ver.read_text().splitlines() if x.strip()]
    by_id = {json.loads(x)["id"]: x for x in lines.read_text().splitlines() if x.strip()}
    return [ComboLine.from_json(json.loads(by_id[i])) for i in ok[:k] if i in by_id]


ap = argparse.ArgumentParser()
ap.add_argument("--pairs", nargs="+", required=True, help="deck0:deck1 (deck0 goes first)")
ap.add_argument("--games", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-turns", type=int, default=6)
ap.add_argument("--budget", type=int, default=4000)
ap.add_argument("--model", default="opencode-go/muse-spark-1.3-contributor")
ap.add_argument("--out", type=Path, default=Path("data/demos/games"))
args = ap.parse_args()
engine, names = Engine(), CardNames()
args.out.mkdir(parents=True, exist_ok=True)
for pair in args.pairs:
    a, b = pair.split(":")
    decks = ((a, load_ydk(Path(f"decks/pool/{a}.ydk"))), (b, load_ydk(Path(f"decks/pool/{b}.ydk"))))
    ex = {a: verified_examples(a), b: verified_examples(b)}
    for g in range(args.games):
        seed = args.seed + g
        print(f"== {a} vs {b}, seed {seed}", flush=True)
        rec = play_game(engine, decks, seed, lambda p: _llm.complete(p, model=args.model, system=TURN_SYSTEM, timeout=900),
                        names, max_turns=args.max_turns, budget=args.budget, examples=ex,
                        log=lambda s: print(s, flush=True))  # fmt: skip
        with (args.out / f"{a}-vs-{b}.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"   winner={rec['winner']} turns={rec['end_turn']} choices={len(rec['choices'])} "
              f"llm={rec['llm_calls']} {rec['seconds']}s lp={rec['lp']}", flush=True)  # fmt: skip
