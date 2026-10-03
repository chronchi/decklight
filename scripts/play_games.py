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
from ygo_zero.demos.verify import plan_from_state
from ygo_zero.engine.duel import Engine
from ygo_zero.infra import remote as _remote


def _remote_plan_fn(engine, config, prefix, line, deck, budget):
    """Verify a turn plan on the games session box; fall back to local search on any error."""
    try:
        return _remote.remote_plan_from_state(config, prefix, line, deck, budget, session="games")
    except Exception as e:  # noqa: BLE001 - remote failure must not kill the game
        print(f"remote plan failed, falling back to local: {e}", flush=True)
        return plan_from_state(engine, config, prefix, line, deck, budget=budget)


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
ap.add_argument("--max-seconds", type=float, default=5400, help="stop at the next turn boundary after this")
ap.add_argument("--variant", default=None, help="muse reasoning effort (minimal|low|high|max; default: provider's)")
ap.add_argument("--remote", action="store_true", default=False,
                help="verify turn plans on the games session box (falls back to local)")
ap.add_argument("--interrupts", type=int, default=3,
                help="LLM interruption asks per turn for the responding player (0 = old random behaviour)")
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
        rec = play_game(engine, decks, seed, lambda p: _llm.complete(p, model=args.model, system=TURN_SYSTEM, timeout=900, variant=args.variant),
                        names, max_turns=args.max_turns, max_seconds=args.max_seconds, budget=args.budget, examples=ex,
                        plan_fn=_remote_plan_fn if args.remote else None, interrupts=args.interrupts,
                        log=lambda s: print(s, flush=True))  # fmt: skip
        with (args.out / f"{a}-vs-{b}.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"   winner={rec['winner']} turns={rec['end_turn']} choices={len(rec['choices'])} "
              f"llm={rec['llm_calls']} {rec['seconds']}s lp={rec['lp']}", flush=True)  # fmt: skip
