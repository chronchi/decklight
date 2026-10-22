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
where = Counter(); offered = Counter(); prev = Counter(); nturn1 = Counter()
for g in G.values():
    if not g.get("s_first", g["s_seat"] == 0): continue
    s = g["s_seat"]; env = CppEnv(eng, S._config(g), True); obs = env.reset(); last = None; ns = 0
    for k, c in enumerate(g["choices"]):
        if obs is None or int(obs["global"][2]) > 1: break
        if int(env.player) == s:
            o = _opts(obs); mx = [i for i, (kd, code) in enumerate(o) if kd == 6 and code == MAXX]
            key = (int(obs["global"][3]), int(obs["global"][21]), int(obs["global"][16+6-6]) if False else len(obs["chain"]) and int(obs["global"][20]))
            if mx:
                offered[key] += 1
                if int(c) in mx:
                    where[key] += 1; prev[last] += 1; nturn1[ns] += 1
            kd, code = o[int(c)]; last = (kd, code); ns += 1
        obs = env.step(int(c))
print("key=(phase, msg, chainlen)")
for k in sorted(offered): print(k, "offered", offered[k], "fired", where[k])
print("move before fire:", prev.most_common(10))
print("S-decision index of fire:", sorted(nturn1.items()))
