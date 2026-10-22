"""Optimizer's curse check on searched windows: when the policy's pick is PASS, override rate to an ACTIVATE by the
number of ACTIVATE candidates; and the mean score gap of those overrides."""
import json, sys, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/audit")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT/"src"))
import show_divergence as S
from decklight.demos.game import _opts
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import Engine
eng = Engine(); G = S.load(sys.argv[1]); lim = int(sys.argv[2])
by = defaultdict(Counter); gaps = defaultdict(list); sel_se = []
for gi, g in enumerate(G.values()):
    if gi >= lim: break
    log = {e["k"]: e for e in g.get("search", [])}
    if not log: continue
    env = CppEnv(eng, S._config(g), True); obs = env.reset(); last = max(log)
    for k, c in enumerate(g["choices"]):
        if obs is None or k > last: break
        if k in log:
            e = log[k]; o = _opts(obs); cands = e["cands"]
            kinds = [o[x][0] for x in cands]
            if o[e["played"]][0] == 11 and e["kind"] == "window":
                nact = sum(kd == 6 for kd in kinds)
                means = [st.mean(x) for x in e["samples"]]
                b = by[min(nact, 3)]; b["n"] += 1
                if o[e["chosen"]][0] == 6:
                    b["fire"] += 1
                    gaps[min(nact, 3)].append(means[cands.index(e["chosen"])] - means[cands.index(e["played"])])
                for x in e["samples"]:
                    if len(x) > 1: sel_se.append(st.stdev(x) / len(x) ** 0.5)
        obs = env.step(int(c))
for n in sorted(by):
    b = by[n]; gp = gaps[n]
    print(f"policy=PASS, {n} ACTIVATE cands: searched {b['n']}, search fires {b['fire']} ({b['fire']/b['n']:.2f}); median gap of fires {st.median(gp) if gp else float('nan'):.3f}")
print("median per-candidate SE of the mean score:", round(st.median(sel_se), 3))
