"""Where do two policies play differently on the same deal? For raw search_match runs of policy A and B on the same
deals (scripts/vast_search_match.sh arms), replay each deal to the first decision where our seat's choices differ and
print the state, A's move and B's move, and both games' results.

    python scripts/audit/show_divergence.py runs/<job>/base runs/<job>/s100 [--n 10] [--seed 0] [--seat second]
        [--stats] [--only-flips] [--deck-dir DIR] [--cards N]

--stats prints whole-data counts instead: share of deals where our seat diverges, the turn of the first divergence, the
option-kind pair (A <- B) and the outcome pattern (W/L for A and B) of diverging deals. --only-flips keeps deals whose
result changed. --deck-dir looks deck names up there first, then decks/gauntlet, then decks/pool (non-gauntlet runs,
e.g. X4b). --stats with --cards N also prints the N most common (A pick <- B pick) card-level labels, overall and
on deals where B won and A lost. Laptop-safe: engine replay only.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import hand_play as H

from decklight.demos.game import _opts, describe_state, hidden_card, option_desc


def load(d: str) -> dict:
    out = {}
    for f in glob.glob(f"{d}/games-*.jsonl"):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                out[(tuple(g["decks"]), g["seed"], g["s_seat"])] = g
    return out


DECK_DIRS = [ROOT / "decks" / "gauntlet", ROOT / "decks" / "pool"]


def _config(g: dict):
    """The deal as search_match builds it: the lists by seat (record order), the game seed."""
    from decklight.decks import load_ydk
    from decklight.engine.duel import DuelConfig

    def path(n: str) -> Path:
        for d in DECK_DIRS:
            if (d / f"{n}.ydk").is_file():
                return d / f"{n}.ydk"
        raise SystemExit(f"deck {n} not in {[str(d) for d in DECK_DIRS]}")

    d0, d1 = (load_ydk(path(n)) for n in g["decks"])
    return DuelConfig(d0, d1, seed=g["seed"])


def first_div(a: dict, b: dict) -> int | None:
    """Index of the first decision where the games differ (both games are identical before it)."""
    for k, (x, y) in enumerate(zip(a["choices"], b["choices"], strict=False)):
        if x != y:
            return k
    return None


def _label(obs, i: int, nn: dict) -> str:
    k, code = _opts(obs)[i]
    nm = nn.get(code, "") if code else hidden_card(obs, i)
    return f"{H._kind(k)} {nm} {option_desc(obs, i)!r}".strip()


def state_at(g: dict, k: int, engine):
    from decklight.engine.cpp_env import CppEnv

    env = CppEnv(engine, _config(g), True)
    obs = env.reset()
    for c in g["choices"][:k]:
        obs = env.step(int(c))
    return obs, int(env.player)


def res(g: dict) -> str:
    if g.get("cut") or g.get("truncated"):
        return "cut"
    win = g["win"] if "win" in g else g.get("winner") == g["s_seat"]  # --opp ckpt/self records carry winner only
    return "W" if win else "L"


def went_first(g: dict) -> bool:
    return bool(g["s_first"]) if "s_first" in g else g["s_seat"] == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seat", choices=["first", "second", "any"], default="any")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--only-flips", action="store_true")
    ap.add_argument("--deck-dir", type=Path, default=None)
    ap.add_argument("--cards", type=int, default=0)
    a = ap.parse_args()
    from decklight.demos.names import CardNames
    from decklight.engine.duel import Engine

    if a.deck_dir is not None:
        DECK_DIRS.insert(0, a.deck_dir)
    A, B = load(a.a), load(a.b)
    keys = sorted(set(A) & set(B))
    if not keys:
        print("no common deals")
        return 1
    if a.seat != "any":
        keys = [k for k in keys if went_first(A[k]) == (a.seat == "first")]
    nn, engine = CardNames().names, Engine()
    div = []
    for key in keys:
        ga, gb = A[key], B[key]
        k = first_div(ga, gb)
        if k is None:
            continue
        if ga["actors"][k] != ga["s_seat"]:
            continue  # the opponent diverged first (only possible via different RNG): not our decision
        if a.only_flips and res(ga) == res(gb):
            continue
        div.append((key, k))
    print(f"{len(keys)} common deals, {len(div)} where our seat's play diverges"
          f" ({len(div) / len(keys):.1%})")  # fmt: skip
    if a.stats:
        turn, kinds, pat, cards, cards_flip = Counter(), Counter(), Counter(), Counter(), Counter()
        for key, k in div:
            ga, gb = A[key], B[key]
            obs, _ = state_at(ga, k, engine)
            o = _opts(obs)
            turn[int(obs["global"][2])] += 1
            kinds[(H._kind(o[int(ga["choices"][k])][0]), H._kind(o[int(gb["choices"][k])][0]))] += 1
            pat[(res(ga), res(gb))] += 1
            if a.cards:
                lab = (_label(obs, int(ga["choices"][k]), nn), _label(obs, int(gb["choices"][k]), nn))
                cards[lab] += 1
                if (res(ga), res(gb)) == ("L", "W"):
                    cards_flip[lab] += 1
        print("turn of first divergence:", dict(sorted(turn.items())))
        print("results A/B on diverging deals:", dict(pat.most_common()))
        print("option kinds (A <- B):")
        for (x, y), v in kinds.most_common(15):
            print(f"  {v:5d} {v / len(div):5.1%}  {x} <- {y}")
        for title, c in (("card-level picks (A <- B):", cards), ("card-level picks on A L -> B W flips:", cards_flip)):
            if a.cards:
                print(title)
                for (x, y), v in c.most_common(a.cards):
                    print(f"  {v:5d}  {x}  <-  {y}")
        return 0
    random.Random(a.seed).shuffle(div)
    for key, k in div[: a.n]:
        ga, gb = A[key], B[key]
        obs, _ = state_at(ga, k, engine)
        print(f"== {key[0][0]} vs {key[0][1]} seed {key[1]} (ours: seat {key[2]}, going "
              f"{'first' if went_first(ga) else 'second'}) decision {k}: A {res(ga)}  B {res(gb)}")  # fmt: skip
        print("\n".join("   " + s for s in describe_state(obs, nn).splitlines()))
        print(f"   A: {_label(obs, int(ga['choices'][k]), nn)}")
        print(f"   B: {_label(obs, int(gb['choices'][k]), nn)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
