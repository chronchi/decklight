"""Export a search_match game record as an EDOPro replay (.yrp) to watch the game in EDOPro (engine only; laptop-safe).

    uv run python scripts/export_yrp.py runs/<job>/<variant>/games-0.jsonl --idx 12 --out docs/showcase/g1.yrp \
        --name0 "Our AI" --name1 "ygo-agent"

Replays the record's choices in the Python reference env (PyEnv) to get the raw core responses, writes the .yrp
(src/ygo_zero/demos/yrp.py), then rebuilds the duel from the file the way EDOPro does and checks the core gives the
same message stream. Exit 2 if the replay does not reproduce the record's winner or the file does not verify.
In EDOPro: the file goes in EDOPro's replay/ folder (--edopro does it), then Replay -> pick it -> Start (EDOPro
re-simulates the duel with its own core and scripts; --edopro checks those reproduce the game).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.decks import load_ydk
from ygo_zero.demos.yrp import edopro_engine, to_yrp, verify
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.engine.env import PyEnv


def deck_path(name: str) -> Path:
    p = Path(name)  # an explicit path (e.g. an edopro_host human.ydk) loads directly
    if p.is_file():
        return p
    if p.suffix != ".ydk" and p.with_suffix(".ydk").is_file():
        return p.with_suffix(".ydk")
    for d in ("pool", "gauntlet", "newcards"):
        p = ROOT / "decks" / d / f"{name}.ydk"
        if p.is_file():
            return p
    sys.exit(f"deck {name} not found in decks/pool|gauntlet|newcards")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", type=Path, help="a search_match games-*.jsonl")
    ap.add_argument("--idx", type=int, required=True, help="game_idx of the record")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name0", default="Player 1", help="name shown for seat 0 (goes first; max 19 chars)")
    ap.add_argument("--name1", default="Player 2")
    ap.add_argument("--deck0", type=Path, default=None, help="override .ydk path for seat 0's deck")
    ap.add_argument("--deck1", type=Path, default=None, help="override .ydk path for seat 1's deck")
    ap.add_argument("--edopro", type=Path, default=None,
                    help="a local EDOPro install (ProjectIgnis/): also verify with its core and scripts, then copy the "
                    "file into its replay/ folder")  # fmt: skip
    a = ap.parse_args()
    rec = next(
        (json.loads(x) for x in a.games.read_text().splitlines() if json.loads(x)["game_idx"] == a.idx), None
    )
    if rec is None:
        sys.exit(f"no game_idx {a.idx} in {a.games}")
    d0, d1 = rec["decks"]
    cfg = DuelConfig(load_ydk(a.deck0 or deck_path(d0)), load_ydk(a.deck1 or deck_path(d1)),
                      seed=rec["seed"])
    engine = Engine()
    env = PyEnv.replay(engine, cfg, rec["choices"])
    if rec.get("truncated") or not env.done or env.winner != rec["winner"]:
        print(f"ALERT game {a.idx}: replay done={env.done} winner={env.winner}, record winner {rec['winner']}",
              file=sys.stderr)  # fmt: skip
        sys.exit(2)
    data = to_yrp(cfg, env.duel.responses, (a.name0, a.name1))
    err = verify(engine, data, env.duel)
    if err:
        print(f"ALERT game {a.idx}: the .yrp does not reproduce the game: {err}", file=sys.stderr)
        sys.exit(2)
    if a.edopro is not None:
        err = verify(edopro_engine(a.edopro), data, env.duel)
        if err:
            print(f"ALERT game {a.idx}: with the EDOPro install's core and scripts: {err}", file=sys.stderr)
            sys.exit(2)
        (a.edopro / "replay" / a.out.name).write_bytes(data)
        print(f"verified with {a.edopro} and copied to its replay/ folder")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    print(
        f"wrote {a.out} ({len(data)} bytes, {len(env.duel.responses)} responses, winner seat {env.winner}; verified)"
    )


if __name__ == "__main__":
    main()
