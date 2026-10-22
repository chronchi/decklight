import json, sys, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/audit")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT/"src"))
import show_divergence as S
from decklight.demos.game import _opts
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import Engine
eng = Engine(); G = S.load(sys.argv[1]); MAXX = 23434538; z = []
for g in G.values():
    if not g.get("s_first"): continue
    log = {e["k"]: e for e in g.get("search", [])}; s = g["s_seat"]
    env = CppEnv(eng, S._config(g), True); obs = env.reset()
    for k, c in enumerate(g["choices"]):
        if obs is None or int(obs["global"][2]) > 1: break
        if k in log:
            e = log[k]; o = _opts(obs)
            if o[e["chosen"]] == (6, MAXX) and e["chosen"] != e["played"] and e["played"] in e["cands"]:
                a = e["samples"][e["cands"].index(e["chosen"])]; b = e["samples"][e["cands"].index(e["played"])]
                d = [x - y for x, y in zip(a, b)]
                z.append((round(st.mean(d), 3), round(st.mean(d) / (st.stdev(d) / len(d) ** 0.5 + 1e-9), 2)))
        obs = env.step(int(c))
print(len(z), "Maxx C overrides: (paired gap, gap/SE)", sorted(z, key=lambda x: x[1]))
print("share with gap >= 2 SE:", sum(t >= 2 for _, t in z) / len(z))
