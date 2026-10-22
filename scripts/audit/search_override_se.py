"""How many search overrides are statistically real? For hppo games (search_match logs with per-candidate rollout
samples, `search` entries), every override (chosen != played = the policy's choice) gets its paired gap
mean(x_chosen - x_played) over the shared sample worlds and gap/SE. Split by option kinds (activation over pass, ...),
duplicate candidates (same kind and card as the policy's), window vs own turn, and by turn bucket.

    python scripts/audit/search_override_se.py runs/<job>/hppo [--max-games N]
Laptop-safe: engine replay only (to name the option kinds).
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import show_divergence as S  # noqa: E402

from decklight.demos.game import _opts  # noqa: E402


def gap_se(a: list[float], b: list[float]) -> tuple[float, float]:
    d = [x - y for x, y in zip(a, b, strict=True)]
    m = st.mean(d)
    se = (st.stdev(d) / len(d) ** 0.5) if len(d) > 1 else float("inf")
    return m, (m / se if se > 1e-9 else (99.0 if m > 0 else 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--max-games", type=int, default=0)
    a = ap.parse_args()
    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import Engine

    eng = Engine()
    G = list(S.load(a.run).values())
    if a.max_games:
        G = G[: a.max_games]
    rows = []
    n_entries = 0
    for g in G:
        log = {e["k"]: e for e in g.get("search", [])}
        if not log:
            continue
        env = CppEnv(eng, S._config(g), True)
        obs = env.reset()
        for k, c in enumerate(g["choices"]):
            if obs is None:
                break
            e = log.get(k)
            if e is not None:
                n_entries += 1
                if e["chosen"] != e["played"] and e["played"] in e["cands"]:
                    o = _opts(obs)
                    ch, pl = o[e["chosen"]], o[e["played"]]
                    m, t = gap_se(e["samples"][e["cands"].index(e["chosen"])], e["samples"][e["cands"].index(e["played"])])
                    n_act = sum(1 for i in e["cands"] if S.H._kind(o[i][0]) == "ACTIVATE")
                    rows.append({"kind": e["kind"], "turn": e["turn"], "gap": m, "t": t, "dup": ch == pl,
                                 "pair": f"{S.H._kind(ch[0])} <- {S.H._kind(pl[0])}", "n_act": n_act,
                                 "win": g.get("win", 0.0)})
            obs = env.step(int(c))
    if not rows:
        print("ALERT search_override_se: no overrides found")
        return 2
    n = len(rows)
    print(f"{len(G)} games, {n_entries} searched decisions, {n} overrides ({n / n_entries:.1%})")
    print(f"overrides with gap >= 2 SE: {sum(r['t'] >= 2 for r in rows) / n:.1%}; >= 1 SE: "
          f"{sum(r['t'] >= 1 for r in rows) / n:.1%}; median gap {st.median(r['gap'] for r in rows):.3f}, "
          f"median gap/SE {st.median(r['t'] for r in rows):.2f}")
    print(f"duplicate-candidate overrides (same kind+card as the policy's): {sum(r['dup'] for r in rows)} "
          f"({sum(r['dup'] for r in rows) / n:.1%}) -- pure noise by construction")
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups["pair " + r["pair"]].append(r)
        groups["where " + r["kind"]].append(r)
        groups["turn " + ("1" if r["turn"] == 1 else "2" if r["turn"] == 2 else "3-5" if r["turn"] <= 5 else "6+")].append(r)
    for name, rs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:16]:
        print(f"  {name:28s} n={len(rs):5d}  >=2SE {sum(r['t'] >= 2 for r in rs) / len(rs):5.1%}  "
              f"median gap {st.median(r['gap'] for r in rs):.3f}")
    pp = Counter((r["n_act"]) for r in rows if r["pair"].startswith("ACTIVATE <- PASS"))
    print("ACTIVATE<-PASS overrides by # activation candidates:", dict(sorted(pp.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
