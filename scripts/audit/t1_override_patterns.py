"""Are turn-1 search overrides a learnable pattern? For searched games (search_match logs with per-candidate rollout
samples), every turn-1 override (chosen != played) gets its paired gap/SE. Decisive ones (z >= --z) are keyed by
(deck, kind+card played -> kind+card chosen); duplicate candidates (same kind and card, e.g. zones) are dropped; a key that recurs in several games of the same deck is a pattern a policy
could learn, a key seen once is a per-deal judgement. Also checks direction: does the reverse key also appear as
a decisive override (the search contradicting itself)?

    python scripts/audit/t1_override_patterns.py runs/<job>/<arm> [--z 3] [--max-games N]
Laptop-safe: engine replay only.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import show_divergence as S
from search_override_se import gap_se

from ygo_zero.demos.game import _opts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--max-games", type=int, default=0)
    a = ap.parse_args()
    from ygo_zero.analysis.events import deck_path
    from ygo_zero.decks import load_ydk
    from ygo_zero.demos.names import CardNames
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.engine.duel import DuelConfig, Engine

    nn = CardNames().names
    eng = Engine()
    G = list(S.load(a.run).values())
    if a.max_games:
        G = G[: a.max_games]
    rows = []
    for g in G:
        log = {e["k"]: e for e in g.get("search", []) if e["turn"] == 1}
        if not log:
            continue
        cfg = DuelConfig(*(load_ydk(deck_path(n)) for n in g["decks"]), seed=g["seed"])
        env = CppEnv(eng, cfg, True)
        obs = env.reset()
        for k, c in enumerate(g["choices"]):
            if obs is None or k > max(log):
                break
            e = log.get(k)
            if e is not None and e["chosen"] != e["played"] and e["played"] in e["cands"]:
                o = _opts(obs)
                m, t = gap_se(
                    e["samples"][e["cands"].index(e["chosen"])], e["samples"][e["cands"].index(e["played"])]
                )
                lab = [
                    f"{S.H._kind(o[i][0])} {nn.get(o[i][1], '-') if o[i][1] else '-'}"
                    for i in (e["played"], e["chosen"])
                ]
                rows.append(
                    {
                        "deck": g["decks"][g["s_seat"]],
                        "game": g["game_idx"],
                        "gap": m,
                        "t": t,
                        "dup": o[e["chosen"]] == o[e["played"]],
                        "key": (lab[0], lab[1]),
                    }
                )
            obs = env.step(int(c))
    if not rows:
        print("ALERT t1_override_patterns: no turn-1 overrides found")
        return 2
    dec = [r for r in rows if r["t"] >= a.z and not r["dup"]]
    ndup = sum(r["t"] >= a.z and r["dup"] for r in rows)
    print(
        f"{len(G)} games, {len(rows)} turn-1 overrides, decisive (z >= {a.z}, not a duplicate candidate) {len(dec)}; "
        f"z >= {a.z} duplicates dropped {ndup}"
    )
    games_by = defaultdict(set)
    for r in dec:
        games_by[(r["deck"], r["key"])].add(r["game"])
    allkeys = Counter((r["deck"], r["key"]) for r in rows)
    rec = Counter()
    for r in dec:
        n = len(games_by[(r["deck"], r["key"])])
        rec["1 game" if n == 1 else "2 games" if n == 2 else "3+ games"] += 1
    print("decisive overrides by how many games of that deck show the same key:", dict(rec))
    rev = sum(1 for r in dec if (r["deck"], (r["key"][1], r["key"][0])) in games_by)
    print(f"decisive overrides whose reverse key is also decisive in that deck: {rev} ({rev / len(dec):.1%})")
    # consistency: for a decisive key, how often does the same (deck, played->chosen) situation appear non-decisive
    cons = [len(games_by[k]) / allkeys[k] for k in games_by]
    print(f"per decisive key: decisive share of all its overrides, median {st.median(cons):.2f}")
    kinds = Counter(f"{r['key'][1].split()[0]} <- {r['key'][0].split()[0]}" for r in dec)
    print("decisive by kind:", kinds.most_common(8))
    top = sorted(games_by.items(), key=lambda kv: -len(kv[1]))[:15]
    print("most recurrent decisive keys (deck: played -> chosen, games):")
    for (d, (p, c)), gs in top:
        print(f"  {d}: {p} -> {c}  ({len(gs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
