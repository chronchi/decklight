"""Play full games between pool decks with the turn planner + engine verifier (ADR 0004, src/ygo_zero/demos/game.py).

nice -n 19 uv run python scripts/play_games.py --pairs blue-eyes:sky-striker --games 1 --max-turns 4
Records: data/demos/games/<deck0>-vs-<deck1>.jsonl (append), one game per line.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from ygo_zero.decks import load_ydk
from ygo_zero.demos import llm as _llm
from ygo_zero.demos.game import TURN_SYSTEM, build_opening, find_openings, play_game, rank_openings
from ygo_zero.demos.names import CardNames
from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.verify import plan_from_state
from ygo_zero.engine.duel import Engine
from ygo_zero.infra import remote as _remote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_lines import load_examples


def _remote_plan_fn(engine, config, prefix, line, deck, budget):
    """Verify a turn plan on the games session box; fall back to local search on any error."""
    try:
        return _remote.remote_plan_from_state(config, prefix, line, deck, budget, session="games")
    except Exception as e:  # noqa: BLE001 - remote failure must not kill the game
        print(f"remote plan failed, falling back to local: {e}", flush=True)
        return plan_from_state(engine, config, prefix, line, deck, budget=budget)


def verified_examples(deck: str, k: int = 2) -> list[ComboLine]:
    """This deck's strongest verified lines from every source (else the strongest of other decks)."""
    return load_examples(deck, k)


ap = argparse.ArgumentParser()
ap.add_argument("--pairs", nargs="+", required=True, help="deck0:deck1 (deck0 goes first)")
ap.add_argument("--games", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-turns", type=int, default=6)
ap.add_argument("--budget", type=int, default=4000)
ap.add_argument("--model", default="opencode/muse-spark-1.3-contributor-free")
ap.add_argument("--out", type=Path, default=Path("data/demos/games"))
ap.add_argument("--max-seconds", type=float, default=5400, help="stop at the next turn boundary after this")
ap.add_argument("--variant", default=None, help="muse reasoning effort for turn plans (default: provider's)")
ap.add_argument("--interrupt-variant", default="minimal", help="muse reasoning effort for interruption asks")
ap.add_argument("--remote", action="store_true", default=False,
                help="verify turn plans on the games session box (falls back to local)")
ap.add_argument("--interrupts", type=int, default=3,
                help="LLM interruption asks per turn for the responding player (0 = old random behaviour)")
ap.add_argument("--from-line", default=None,
                help="verified records file/dir for going-second openings: replay a verified turn-1 line "
                     "of deck A as turn 1 instead of planning it (deck B then plays turn 2 against the board)")
ap.add_argument("--from-index", type=int, default=0,
                help="which candidate opening record to replay (candidates ranked by biggest end board)")
ap.add_argument("--random-records", type=int, default=0,
                help="sample N candidate opening records (seeded) and cycle through them across games")
ap.add_argument("--opening-frac", type=float, default=0.0,
                help="probability a game starts from a going-second opening (seeded); without --from-line "
                     "the candidates come from data/demos. --from-line alone (frac 0) opens every game.")
args = ap.parse_args()
engine, names = Engine(), CardNames()
args.out.mkdir(parents=True, exist_ok=True)
DEMOS = Path(__file__).resolve().parents[1] / "data" / "demos"
rng = random.Random(args.seed)


def _pool_for(deck_a: str) -> list[dict]:
    """Candidate opening records of deck A: --from-line (file or dir) or the data/demos verified records."""
    if args.from_line is None:
        return find_openings(deck_a, DEMOS)
    p = Path(args.from_line)
    if p.is_dir():
        return find_openings(deck_a, p)
    return rank_openings([json.loads(x) for x in p.read_text().splitlines() if x.strip()], deck_a)


for pair in args.pairs:
    a, b = pair.split(":")
    decks = ((a, load_ydk(Path(f"decks/pool/{a}.ydk"))), (b, load_ydk(Path(f"decks/pool/{b}.ydk"))))
    ex = {a: verified_examples(a), b: verified_examples(b)}
    pool = _pool_for(a) if (args.from_line is not None or args.opening_frac > 0) else []
    spread = rng.sample(pool, min(args.random_records, len(pool))) if args.random_records > 0 and pool else []
    for g in range(args.games):
        seed = args.seed + g
        opening, opening_id = None, None
        use_opening = bool(pool) and (
            rng.random() < args.opening_frac if args.opening_frac > 0 else args.from_line is not None
        )
        if use_opening:
            assert pool
            cands = spread if spread else pool
            start = 0 if spread else args.from_index % len(pool)
            for j in range(len(cands)):
                rec0 = cands[(start + j) % len(cands)]
                built = build_opening(engine, rec0, decks[1][1], seed=seed)
                if built is not None:
                    opening, opening_id = built, rec0["line_id"]
                    break
            if opening is None:
                print("   no valid opening record, normal start", flush=True)
            else:
                print(f"   opening={opening_id}", flush=True)
        print(f"== {a} vs {b}, seed {seed}", flush=True)
        rec = play_game(engine, decks, seed, lambda p: _llm.complete(p, model=args.model, system=TURN_SYSTEM, timeout=900, variant=args.variant),
                        names, max_turns=args.max_turns, max_seconds=args.max_seconds, budget=args.budget, examples=ex,
                        plan_fn=_remote_plan_fn if args.remote else None, interrupts=args.interrupts,
                        interrupt_fn=lambda p: _llm.complete(p, model=args.model, system=TURN_SYSTEM, timeout=300, variant=args.interrupt_variant),
                        opening=opening, log=lambda s: print(s, flush=True))  # fmt: skip
        rec["opening"] = opening_id
        with (args.out / f"{a}-vs-{b}.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"   winner={rec['winner']} turns={rec['end_turn']} choices={len(rec['choices'])} "
              f"llm={rec['llm_calls']} {rec['seconds']}s lp={rec['lp']}", flush=True)  # fmt: skip
