"""Going-first turn-1 boards per deck: ours (s_seat 0) vs ygo-agent (s_seat 1, it sits in seat 0) from raw
search_match games vs ygoagent. Field = seat 0's own field cards at its last turn-1 decision.
Usage: python scripts/audit/turn1_boards_vs_ya.py runs/<final-card>/raw"""
import json, glob, sys, statistics
sys.path.insert(0, "scripts/audit"); sys.path.insert(0, "src")
import show_divergence as S
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import Engine
from ygo_zero.eval.turn1 import my_field
eng = Engine()
G = S.load(sys.argv[1])
def t1(g):
    env = CppEnv(eng, S._config(g), True); obs = env.reset(); last = 0; n = 0
    for c in g["choices"]:
        if obs is None or int(obs["global"][2]) > 1: break
        if env.player == 0: last = my_field(obs); n += 1
        obs = env.step(int(c))
    return last, n
by = {}
for (decks, seed, ss), g in G.items():
    f, n = t1(g); by.setdefault(decks[0], {0: [], 1: []})[ss].append((f, n))
M = statistics.mean
ours = [x for v in by.values() for x in v[0]]; ya = [x for v in by.values() for x in v[1]]
print("going-first turn-1 field: ours", round(M(f for f, _ in ours), 2), "n", len(ours), "| ygo-agent", round(M(f for f, _ in ya), 2), "n", len(ya))
print("turn-1 decisions: ours", round(M(n for _, n in ours), 1), "ygo-agent", round(M(n for _, n in ya), 1))
for k, w in ((0, "ours"), (1, "ya")):
    xs = sorted(f for v in by.values() for f, _ in v[k]); print(w, "field quartiles", xs[len(xs)//4], xs[len(xs)//2], xs[3*len(xs)//4], "zero-board", round(sum(x <= 1 for x in xs)/len(xs), 3))
rows = sorted((M(f for f, _ in v[0]) - M(f for f, _ in v[1]), d, M(f for f, _ in v[0]), M(f for f, _ in v[1])) for d, v in by.items() if v[0] and v[1])
print("decks where ygo-agent > ours:", sum(r[0] < 0 for r in rows), "of", len(rows))
for r in rows[:6] + rows[-4:]: print(f"  {r[1]:18s} ours {r[2]:.2f} ya {r[3]:.2f}")
