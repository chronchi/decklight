"""Why does an RL checkpoint lose to ygo-agent more than its base? Paired game analysis (2026-10-08, rl5).

    uv run python scripts/audit/rl_ga_compare.py runs/<rl5ga run> --base it0 [--vs it200 ...] [--render 6]

RL runs save every eval game (train_rl, since 2026-10-08): compare two iterations of one run with
    uv run python scripts/audit/rl_ga_compare.py runs/<run>/rl/games --base iter-0000/vs_ygoagent \
        --vs iter-0200/vs_ygoagent [--render 6]        # or .../vs_bc for the games vs the base
then `rl_ga_divergence.py` (same args, --tag) and `rl_chain_usage.py` for the decision-level view.

Input: `<run>/<tag>/games.jsonl` + `games/*/state.json` from `scripts/vast_ga_compare.sh` (each checkpoint vs
ygo-agent on the same schedule; ygo-agent and our policy are greedy, so the same deal gives the same game until our
policy's choices differ). Games pair by (deck_a, deck_b, seed, a_first). Per compared tag it prints:
- win rates and the paired transition table vs the base (WW / WL / LW / LL, exact McNemar p);
- for the flipped games (WL = base won, tag lost; LW = the reverse): where the two games first diverge (turn, whose
  turn, decision index), our field size at the end of our first turn in each, game length;
- per seat and per our deck: win rate base vs tag;
- the same turn-1 field and length statistics over all games (did the opening combo change?).
With --render K: writes K WL pairs as readable logs (render_games format) under <run>/flips/<tag>/.
Engine only (replay), no model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from hand_play import _config
from render_games import render_one

from ygo_zero.demos.names import CardNames
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import Engine

FIELD_LOCS = (3, 4)  # obs card column 1: 3 mzone, 4 szone (same measure as playtest.my_field)


def load(run: Path, tag: str) -> dict:
    """key -> {game dir, a_win, turns, decisions, state}; seed-0 games ("<tag>-s0") and the rest ("<tag>-ga")."""
    out = {}
    for raw in (run / tag / "games.jsonl").read_text().splitlines():
        g = json.loads(raw)
        key = (g["deck_a"], g["deck_b"], g["seed"], g["a_first"])
        if key in out:
            raise SystemExit(f"{tag}: duplicate game {key}")
        if "choices" in g:  # train_rl eval format (rl.evaluate.save_games): the replay is inline
            g["state"] = {"decks": g["decks"], "seed": g["seed"], "choices": g["choices"], "a_seat": g["a_seat"],
                          "winner": None}
            g["path"] = None
        else:  # gauntlet_games format: one state.json per game
            g["path"] = run / tag / "games" / g["game"] / "state.json"
        out[key] = g
    if not out:
        raise SystemExit(f"{tag}: 0 games")
    return out


def fields(engine: Engine, st: dict) -> dict:
    """Replay: our field at the end of each of our turns (by our turn index), the opponent's at the end of theirs.
    Fills st["actors"] / st["turns"] when the record lacks them (train_rl eval format)."""
    seat = int(st["a_seat"])
    env = CppEnv(engine, _config(st))
    obs = env.reset()
    mine: dict[int, int] = {}
    theirs: dict[int, int] = {}
    actors, turns = [], []
    for c in st["choices"]:
        p, turn = env.player, int(obs["global"][2])
        actors.append(int(p))
        turns.append(turn)
        cm, mm = obs["cards"], obs["card_mask"]
        n_own = int(sum(1 for j in np.flatnonzero(mm) if int(cm[j, 1]) in FIELD_LOCS and int(cm[j, 3]) == 0))
        (mine if p == seat else theirs)[turn] = n_own  # last value seen in that turn = end-of-turn proxy
        obs = env.step(c)
        if obs is None:
            break
    st.setdefault("actors", actors)
    st.setdefault("turns", turns)
    return {"mine": mine, "theirs": theirs}


def first_turn_field(f: dict, a_first: bool) -> int:
    t = 1 if a_first else 2
    return f["mine"].get(t, 0)


def divergence(sa: dict, sb: dict) -> tuple[int, int, int] | None:
    """(decision index, turn, actor) of the first differing choice; None if identical."""
    ca, cb = sa["choices"], sb["choices"]
    for i in range(min(len(ca), len(cb))):
        if ca[i] != cb[i]:
            return i, int(sa["turns"][i]), int(sa["actors"][i])
    if len(ca) != len(cb):
        i = min(len(ca), len(cb))
        src = sa if len(ca) > len(cb) else sb
        return i, int(src["turns"][i]), int(src["actors"][i])
    return None


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def med(xs) -> str:
    return f"{np.median(xs):.1f}" if len(xs) else "-"


def render(engine: Engine, names: CardNames, path: Path, header: str) -> str:
    """render_games format (public log + every non-forced decision); writes game.txt next to state.json."""
    render_one(path, True, names, engine)
    return header + "\n" + (path.parent / "game.txt").read_text()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--base", default="it0")
    ap.add_argument("--vs", nargs="*", default=None, help="tags to compare (default: every other tag dir)")
    ap.add_argument("--render", type=int, default=0)
    args = ap.parse_args()
    tags = args.vs or sorted(p.name for p in args.run.iterdir() if (p / "games.jsonl").is_file() and p.name != args.base)
    engine = Engine()
    base = load(args.run, args.base)
    cache: dict = {}

    def st_of(g):
        key = g["path"] or id(g)
        if key not in cache:
            st = dict(g["state"]) if g["path"] is None else json.loads(g["path"].read_text())
            st["fields"] = fields(engine, st)
            cache[key] = st
        return cache[key]

    def summary(tag: str, games: dict) -> None:
        f1 = [first_turn_field(st_of(g)["fields"], g["a_first"]) for g in games.values()]
        f1_first = [first_turn_field(st_of(g)["fields"], True) for g in games.values() if g["a_first"]]
        wr = np.mean([g["a_win"] for g in games.values()])
        wr1 = np.mean([g["a_win"] for g in games.values() if g["a_first"]])
        wr2 = np.mean([g["a_win"] for g in games.values() if not g["a_first"]])
        print(f"{tag:>6}: n {len(games)} wr {wr:.3f} (first {wr1:.3f}, second {wr2:.3f}) | our field after our "
              f"1st turn: mean {np.mean(f1):.2f}, going first {np.mean(f1_first):.2f}, 0-card {np.mean(np.array(f1) == 0):.2f} | "
              f"turns median {med([g['turns'] for g in games.values()])}, decisions median "
              f"{med([g['decisions'] for g in games.values()])}")

    print("== all games")
    summary(args.base, base)
    for tag in tags:
        summary(tag, load(args.run, tag))

    for tag in tags:
        other = load(args.run, tag)
        keys = sorted(set(base) & set(other))
        if len(keys) != len(base) or len(keys) != len(other):
            print(f"WARNING {tag}: {len(keys)} paired of {len(base)}/{len(other)}")
        t = Counter()
        flips = {"WL": [], "LW": []}
        same = 0
        for k in keys:
            a, b = base[k]["a_win"], other[k]["a_win"]
            code = ("W" if a == 1 else "L" if a == 0 else "D") + ("W" if b == 1 else "L" if b == 0 else "D")
            t[code] += 1
            if code in flips:
                flips[code].append(k)
            if st_of(base[k])["choices"] == st_of(other[k])["choices"]:
                same += 1
        print(f"\n== {args.base} -> {tag}: {len(keys)} paired deals; identical games {same} ({same / len(keys):.2f})")
        print("   " + "  ".join(f"{c} {t[c]}" for c in sorted(t)) + f"   McNemar p = {mcnemar(t['WL'], t['LW']):.3f}")
        for code, ks in flips.items():
            if not ks:
                continue
            div = [divergence(st_of(base[k]), st_of(other[k])) for k in ks]
            seat_of = {k: (0 if k[3] else 1) for k in ks}
            where = Counter()
            for k, d in zip(ks, div, strict=True):
                if d is None:
                    where["identical?"] += 1
                    continue
                _, turn, actor = d
                whose = "our" if actor == seat_of[k] else "opp"
                where[f"turn {turn if turn <= 4 else '5+'} ({whose})"] += 1
            fb = [first_turn_field(st_of(base[k])["fields"], k[3]) for k in ks]
            fo = [first_turn_field(st_of(other[k])["fields"], k[3]) for k in ks]
            print(f"   {code} ({len(ks)}): going first {sum(k[3] for k in ks)}, second {sum(not k[3] for k in ks)}")
            print("      first divergence: " + ", ".join(f"{w} {n}" for w, n in sorted(where.items())))
            print(f"      our field after our 1st turn: {args.base} mean {np.mean(fb):.2f} -> {tag} {np.mean(fo):.2f}; "
                  f"bigger {sum(o > b for b, o in zip(fb, fo))}, smaller {sum(o < b for b, o in zip(fb, fo))}, "
                  f"same {sum(o == b for b, o in zip(fb, fo))}")
            print(f"      turns median {args.base} {med([base[k]['turns'] for k in ks])} -> {tag} "
                  f"{med([other[k]['turns'] for k in ks])}")
        per_deck = defaultdict(lambda: [0.0, 0.0, 0])
        for k in keys:
            d = per_deck[k[0]]
            d[0] += base[k]["a_win"]
            d[1] += other[k]["a_win"]
            d[2] += 1
        rows = sorted(per_deck.items(), key=lambda kv: (kv[1][1] - kv[1][0]))
        print("   per our deck (wins base -> tag of n), largest drops first:")
        print("   " + "; ".join(f"{d.split('.')[0]} {int(a)}->{int(b)}/{n}" for d, (a, b, n) in rows[:10]))
        print("   largest gains: " + "; ".join(f"{d.split('.')[0]} {int(a)}->{int(b)}/{n}" for d, (a, b, n) in rows[-5:]))
        if args.render and flips["WL"]:
            names = CardNames()
            od = args.run / "flips" / tag.replace("/", "_")
            od.mkdir(parents=True, exist_ok=True)
            for i, k in enumerate(flips["WL"][: args.render]):
                d = divergence(st_of(base[k]), st_of(other[k]))
                head = (f"deal {k}: we pilot {k[0]} vs ygo-agent {k[1]}, seed {k[2]}, we go "
                        f"{'first' if k[3] else 'second'}; first divergence {d}")
                for tg, g in ((args.base, base[k]), (tag, other[k])):
                    path = g["path"]
                    if path is None:  # inline record: write a state.json for render_games
                        path = od / "states" / f"{i:02d}-{tg.replace('/', '_')}" / "state.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(dict(g["state"], labels=["gold"] * len(g["state"]["choices"]),
                                                        notes=[])))
                    txt = render(engine, names, path,
                                 f"{head}\n{tg}: {'WIN' if g['a_win'] == 1 else 'LOSS'}, {g['turns']} turns")
                    (od / f"{i:02d}-{tg.replace('/', '_')}.txt").write_text(txt)
            print(f"   rendered {min(args.render, len(flips['WL']))} WL pairs to {od}")


if __name__ == "__main__":
    main()
