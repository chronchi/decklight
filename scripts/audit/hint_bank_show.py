"""Read a hint bank before training on it (quality gate 5, NOTEBOOK 2026-10-08 21:05).

    uv run python scripts/audit/hint_bank_show.py runs/<hintgated run>/bank \
        --starts data/cache/dags/starts-train-1k.jsonl [--slots <slots.jsonl>] [--show 10] [--seed 0]

Per banked (start, seat): decks, the slot's league pass rate (with --slots), LP and resources at the start, hinted wins
vs each opponent (from games-*.jsonl), the turn span (cut - end) of each banked win. Then --show K random banked
starts rendered (render_games format: public log + every non-forced decision) to <bank>/show/NN.txt; the header says
from which decision on the game is live (earlier ones are the DAGS prefix from the source game). Engine only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from render_games import render_one

from ygo_zero.demos.names import CardNames
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import Engine
from ygo_zero.eval.turn1 import config_from_dict


def turns_of(engine: Engine, r: dict) -> list[int]:
    """The turn of every banked decision (replay)."""
    env = CppEnv(engine, config_from_dict(r["config"]))
    obs = env.reset()
    out = []
    for c in r["choices"]:
        if obs is None:
            raise SystemExit(f"start {r['start_id']}: banked row ends before its last choice")
        out.append(int(obs["global"][2]))
        obs = env.step(int(c))
    return out


def span(engine: Engine, r: dict) -> tuple[int, int]:
    """(turn at the cut, turn the game ends)."""
    t = turns_of(engine, r)
    k = len(r["prefix"])
    return (t[k] if k < len(t) else t[-1]), t[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bank", type=Path, help="hint_bank --out dir (bank.jsonl + games-*.jsonl)")
    ap.add_argument("--starts", type=Path, default=ROOT / "data/cache/dags/starts-train-1k.jsonl")
    ap.add_argument("--slots", type=Path, default=None, help="dags_hopeless slots.jsonl (pass rate, LP, resources)")
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(x) for x in (args.bank / "bank.jsonl").read_text().splitlines() if x.strip()]
    if not rows:
        raise SystemExit("0 banked rows")
    starts = [json.loads(x) for x in args.starts.read_text().splitlines() if x.strip()]
    slots = {}
    if args.slots:
        for x in args.slots.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                slots[(r["start"], r["seat"])] = r
    games = defaultdict(lambda: {"raw": [0, 0], 0: [0, 0], 1: [0, 0]})
    for f in sorted(args.bank.glob("games-*.jsonl")):
        for x in f.read_text().splitlines():
            if x.strip():
                g = json.loads(x)
                k = "raw" if g["kind"] == "raw" else int(g.get("opp", 0))
                c = games[(int(g["start_id"]), int(g["actor"]))][k]
                c[0] += bool(g["actor_win"])
                c[1] += 1
    by = defaultdict(list)
    for r in rows:
        by[(int(r["start_id"]), int(r["actor_seat"]))].append(r)
    engine, names = Engine(), CardNames()
    print(f"{len(rows)} banked games on {len(by)} (start, seat) slots; decks (actor side): "
          f"{Counter(starts[s]['decks'][seat] for s, seat in by).most_common(10)}")
    print(f"{'start':>5} {'seat':>4} {'actor deck':<24} {'opp deck':<24} {'p_league':>8} {'LP me/opp':>11} "
          f"{'res':>4} {'raw':>4} {'vs0':>4} {'vs1':>4} {'cut-end turns':>14}")
    for (s, seat), rs in sorted(by.items()):
        sl = slots.get((s, seat), {})
        cnt = sl.get("cnt", {"me": {}, "opp": {}})
        res = sum(cnt["me"].get(k, 0) for k in ("hand", "mzone", "szone")) - sum(
            cnt["opp"].get(k, 0) for k in ("hand", "mzone", "szone"))
        g = games[(s, seat)]
        ds = starts[s]["decks"]
        p = f"{sl['w']:.0f}/{sl['n']}" if sl else "-"
        lp = f"{sl.get('lp_me', '?')}/{sl.get('lp_opp', '?')}"
        turns = ",".join("{}-{}".format(*span(engine, r)) for r in rs)
        print(f"{s:>5} {seat:>4} {ds[seat]:<24} {ds[1 - seat]:<24} {p:>8} {lp:>11} {res:>4} "
              f"{g['raw'][0]}/{g['raw'][1]:<2} {g[0][0]}/{g[0][1]:<2} {g[1][0]}/{g[1][1]:<2} {turns:>14}")
    if not args.show:
        return
    od = args.bank / "show"
    od.mkdir(exist_ok=True)
    pick = random.Random(args.seed).sample(sorted(by), min(args.show, len(by)))
    for i, (s, seat) in enumerate(pick):
        r = by[(s, seat)][0]
        ds = starts[s]["decks"]
        st = {"config": r["config"], "decks": ds, "seed": r["config"]["seed"], "choices": r["choices"],
              "actors": r["actors"], "turns": (tt := turns_of(engine, r)), "labels": ["gold"] * len(r["choices"]), "notes": [],
              "meta": {"policy_seat": seat, "opponent": f"opp{r.get('opp', 0)}", "match": f"start {s}",
                       "winner": r["winner"], "policy_won": r["winner"] == seat, "turns": tt[-1]}}
        sp = od / f"{i:02d}" / "state.json"
        sp.parent.mkdir(exist_ok=True)
        sp.write_text(json.dumps(st))
        render_one(sp, True, names, engine)
        head = (f"start {s} seat {seat}: {ds[seat]} (actor, base_r3 + search) vs {ds[1 - seat]} (opp{r.get('opp', 0)}); "
                f"DAGS prefix = decisions 0..{len(r['prefix']) - 1}, live from decision {len(r['prefix'])} "
                f"(start turn {r.get('turn_start')}); hinted wins {games[(s, seat)]}\n")
        (od / f"{i:02d}.txt").write_text(head + (sp.parent / "game.txt").read_text())
    print(f"rendered {len(pick)} banked starts to {od}")


if __name__ == "__main__":
    main()
