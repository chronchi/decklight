"""Python oracle for the A2a view frame (mirrors rust/decklight-view/src/bin/view_dump.rs).

    uv run python scripts/app/view_dump.py --pairs A:B C:D --games N --seed S --choice-seed C --out FILE

Plays `PyEnv(..., auto_single=False)` games with the splitmix64 choice stream of `play_dump.py` and writes one
JSON line per decision, then the end frame and the reveal-all replay frame, exactly as the Rust binary does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.app.duel_hash import SplitMix
from scripts.app.play_dump import resolve_deck


def run(args) -> int:
    from decklight.decks import load_ydk
    from decklight.engine import messages as M
    from decklight.engine.duel import DuelConfig, Engine
    from decklight.engine.env import PyEnv
    from decklight.web.cues import build_cues
    from decklight.web.stats import DuelStats
    from decklight.web.view import build_events, build_prompt, build_view

    if args.pairs:
        pairs = [(resolve_deck(a), resolve_deck(b)) for a, b in (p.split(":", 1) for p in args.pairs)]
    elif args.deck0 and args.deck1:
        pairs = [(Path(args.deck0), Path(args.deck1))]
    else:
        raise SystemExit("--pairs or --deck0/--deck1 is required")
    decks = [(load_ydk(a), load_ydk(b)) for a, b in pairs]
    engine = Engine()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    offset = int(args.game_offset or 0)
    total = 0
    with out.open("w", encoding="utf-8") as fh:
        for i in range(args.games):
            g = offset + i
            d0, d1 = decks[g % len(decks)]
            seed = args.seed + g
            env = PyEnv(engine, DuelConfig(d0, d1, seed=seed), auto_single=False)
            env.reset()
            h = g % 2
            rng = SplitMix(args.choice_seed ^ g)
            stats = DuelStats(h)
            cue_sent = 0
            log_sent = 0
            prompt_id = 0
            last_dec = None
            k = 0
            while env.dec is not None:
                if env.dec is not last_dec:
                    last_dec = env.dec
                    prompt_id += 1
                view = build_view(env, h)
                prompt = build_prompt(env, h, prompt_id) if env.player == h else None
                events = build_events(env, h, log_sent)
                cues, cue_sent = build_cues(env, h, since=cue_sent)
                stats.feed_cues(cues)
                stats.note_view(view)
                fh.write(
                    json.dumps(
                        {
                            "g": g,
                            "k": k,
                            "h": h,
                            "view": view,
                            "prompt": prompt,
                            "events": events,
                            "cues": cues,
                            "stats": stats.stats_message(winchance_on=False),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                log_sent = len(env.history)
                n = len(env.opts)
                if n <= 0:
                    raise RuntimeError(f"game {g} decision {k}: no legal option")
                idx = int(rng.next() % n)
                is_chain = isinstance(env.dec.msg, M.SelectChain) and env.player == h
                had = bool(is_chain) and any(o.kind.name == "ACTIVATE" for o in env.opts)
                used = bool(is_chain and 0 <= idx < len(env.opts) and env.opts[idx].kind.name == "ACTIVATE")
                env.step(idx)
                if is_chain:
                    stats.note_chain_window(had, used)
                k += 1
            view = build_view(env, h)
            events = build_events(env, h, log_sent)
            cues, cue_sent = build_cues(env, h, since=cue_sent)
            stats.feed_cues(cues)
            stats.note_view(view)
            fh.write(
                json.dumps(
                    {
                        "g": g,
                        "k": "end",
                        "h": h,
                        "view": view,
                        "prompt": None,
                        "events": events,
                        "cues": cues,
                        "stats": stats.stats_message(winchance_on=False),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            view_all = build_view(env, h, reveal_all=True)
            cues_all, _ = build_cues(env, h, 0, reveal_all=True)
            fh.write(
                json.dumps(
                    {"g": g, "end": True, "view_all": view_all, "cues_all": cues_all},
                    separators=(",", ":"),
                )
                + "\n"
            )
            env.duel.close()
            total += k
            print(f"game {g}: decisions {k}", file=sys.stderr)
    print(f"wrote {total} decisions to {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck0", type=Path, default=None)
    ap.add_argument("--deck1", type=Path, default=None)
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--choice-seed", type=int, default=0)
    ap.add_argument("--game-offset", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data/cache/app/view_py.jsonl")
    a = ap.parse_args(argv)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
