"""For P2 arms: at S's own-turn-1 decisions where a hand trap ACTIVATE is offered, was it taken, searched, scores?"""
import json, sys, glob
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/audit")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT/"src"))
import show_divergence as S
from decklight.demos.game import _opts
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import Engine
HT = {23434538: "MaxxC", 24224830: "CBTG", 14558127: "Ash", 10045474: "Imperm", 97268402: "Veiler", 27204311: "Nibiru", 94145021: "Droll", 59438930: "Ogre", 73642296: "Belle", 44146295:"Mirrorjade"}
eng = Engine()
arm = sys.argv[1]
G = S.load(arm)
st = defaultdict(Counter)
rows = []
for g in G.values():
    if not g.get("s_first", g["s_seat"] == 0): continue
    s = g["s_seat"]
    log = {e["k"]: e for e in g.get("search", [])}
    env = CppEnv(eng, S._config(g), True); obs = env.reset()
    for k, c in enumerate(g["choices"]):
        if obs is None: break
        turn = int(obs["global"][2])
        if turn > 1: break
        if int(env.player) == s:
            o = _opts(obs)
            offered = {HT[code] for (kd, code) in o if kd == 6 and code in HT}
            kd, code = o[int(c)]
            took = HT.get(code) if kd == 6 else None
            for h in offered:
                st[h]["offered"] += 1
                if k in log: st[h]["searched"] += 1
                if took == h:
                    st[h]["taken"] += 1
                    if k in log:
                        st[h]["taken_searched"] += 1
                        e = log[k]
                        means = [sum(x)/len(x) for x in e["samples"]] if e.get("samples") else None
                        rows.append({"deck": g["decks"][s], "seed": g["seed"], "h": h, "cands": [(x, o[x]) for x in e["cands"]], "means": means, "chosen": e["chosen"], "played": e["played"], "c": int(c)})
                    else:
                        st[h]["taken_unsearched"] += 1
                if k in log and log[k]["played"] != log[k]["chosen"]:
                    st[h]["override_here"] += 1
        obs = env.step(int(c))
for h, c in st.items(): print(h, dict(c))
for r in rows[:40]:
    print(json.dumps(r))
