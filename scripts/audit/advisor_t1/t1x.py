import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/audit")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT/"src"))
import show_divergence as S
from decklight.demos.game import _opts
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import Engine
eng = Engine(); G = S.load(sys.argv[1]); MAXX = 23434538
tab = Counter(); deltas = []; confs = []
for g in G.values():
    if not g.get("s_first"): continue
    s = g["s_seat"]; log = {e["k"]: e for e in g.get("search", [])}
    env = CppEnv(eng, S._config(g), True); obs = env.reset()
    for k, c in enumerate(g["choices"]):
        if obs is None or int(obs["global"][2]) > 1: break
        if int(env.player) == s:
            o = _opts(obs)
            mx = [i for i, (kd, code) in enumerate(o) if kd == 6 and code == MAXX]
            if mx:
                if k in log:
                    e = log[k]; cands = e["cands"]
                    means = [sum(x)/len(x) for x in e["samples"]]
                    pm = e["played"] in mx; cm = e["chosen"] in mx; incand = any(x in mx for x in cands)
                    tab[("searched", "pol=Maxx" if pm else "pol=other", "srch=Maxx" if cm else "srch=other", "Maxx in cands" if incand else "not cand")] += 1
                    if incand:
                        bm = max(means[j] for j, x in enumerate(cands) if x in mx)
                        bo = max([means[j] for j, x in enumerate(cands) if x not in mx] or [float('nan')])
                        deltas.append(bm - bo)
                else:
                    tab[("unsearched", "taken=Maxx" if int(c) in mx else "other")] += 1
        obs = env.step(int(c))
for k, v in sorted(tab.items()): print(v, k)
import statistics as st
d = [x for x in deltas if x == x]
print("n", len(d), "mean delta(Maxx - best other)", round(st.mean(d), 4), "median", round(st.median(d), 4), "share>0", round(sum(x > 0 for x in d)/len(d), 3))
